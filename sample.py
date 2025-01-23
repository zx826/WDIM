import torch
import pywt
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.utils.data import SequentialSampler
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from Model import Mamba, MambaConfig
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from loss import Hierarchical_SSIM,PerceptualLoss
from wavelet import DWT

'''训练帮助的函数'''


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):  # 将 EMA 模型逐步推向当前模型。
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():  # TTODO：考虑仅应用于 require_grad 的参数，以避免 pos_embed 的微小数值变化
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):  # 为模型中的所有参数设置 require_grad 标志，需要就可以设置梯度
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):  # 创造一个logger写入日志文件和标准输出
    logging.basicConfig(  # 用默认格式化程序创建 StreamHandler 并将其添加到根日志记录器中，从而完成日志系统的基本配置
        level=logging.INFO,  # 将根记录器级别设置为指定的级别。低于该级别的就不输出了。级别排序：CRITICAL > ERROR > WARNING > INFO > DEBUG
        format='[\033[34m%(asctime)s\033[0m] %(message)s',  # 为处理程序使用指定的格式字符串
        datefmt='%Y-%m-%d %H:%M:%S',  # 使用 time.strftime() 所接受的指定日期/时间格式
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        # 如果指定，这应该是已经创建的处理程序的迭代，以便添加到根日志程序中。
        # 任何没有格式化程序集的处理程序都将被分配给在此函数中创建的默认格式化程序。注意，此参数与 filename 或 stream 不兼容——如果两者都存在，则会抛出 ValueError。
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):  # ADM 的实施中心裁剪。

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def load_checkpoint(checkpoint_path, model, args):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # 加载模型状态
        model.load_state_dict(checkpoint["model"])
        # 更新训练参数
        args.__dict__.update(checkpoint["args"].__dict__)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        return True
    else:
        logging.warning(f"No checkpoint found at {checkpoint_path}")
        return False


'''循环训练'''


def main(args):  # 训练一个新的模型

    assert torch.cuda.is_available(), "至少需要一个GPU"
    print('GPU是否可用：', torch.cuda.is_available())

    # 设置实验文件夹
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}"  # 创建实验文件夹
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # 存储保存的模型检查点
    os.makedirs(checkpoint_dir, exist_ok=True)  # 递归目录创建函数
    logger = create_logger(experiment_dir)  # 上面定义的函数创建记录点
    logger.info(f"Experiment directory created at {experiment_dir}")  # logging.info是输出日志的信息

    # 创造模型
    assert args.image_size % 8 == 0, "图像大小必须能被 8 整除（对于 VAE 编码器）"

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    dwt = DWT()
    config = MambaConfig(d_model=1280, n_layers=28)  # 模型设置d_model=1000, n_layers=20
    model = Mamba(config)
    model = model.to('cuda')

    #vae = AutoencoderKL.from_pretrained(f"sd-vae-ft-{args.vae}").to('cuda')  # 通过model.from_pretrained函数来加载预训练模型
    logger.info(f"WDIM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 设置优化器（使用默认的 Adam betas=（0.9， 0.999） 和 1e-4 的恒定学习率）:
    opt = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0)

    # 设置数据:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

   

    # combined_dataset = ConcatDataset([raw, gt])
    


    val_dataset = ImageFolder(args.data_path_val, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(1),
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )


    # 为训练准备模型:
    


    checkpoint_dir = '/root/autodl-tmp/results/000/checkpoints'


    # 获取最新的 checkpoint 文件路径
    if os.path.exists(checkpoint_dir):
        checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if checkpoint_files:
            last_checkpoint_path = checkpoint_files[-1]
            checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint_path)
        else:
            checkpoint_path = None
    else:
        checkpoint_path = None
    # 加载 checkpoint
    if checkpoint_path:
        success = load_checkpoint(checkpoint_path,model, args)
    else:
        success = False
    if success:
        print("Checkpoint loaded successfully. Continuing training...")
    else:
        print("No checkpoint found. Starting training from scratch...")


    i = 0
    for x, _ in val_loader:
        i = i + 1
        x = x.to('cuda')
        with torch.no_grad():
            # z, _, _ = dwt(z, 3)
            save_image(x, f"/root/autodl-tmp/generate/img{i}.png", nrow=4, normalize=True,value_range=(-1, 1))
            u, H, _ = dwt(x, 3)
            noise = torch.randn_like(u).to('cuda')


            sample, OUT = diffusion.p_sample_loop(
                model, noise.shape, noise, u, H[0], H[1], H[2], clip_denoised=False, model_kwargs=None,
                progress=True, device='cuda')
            HH1 = OUT['h1']
            HH2 = OUT['h2']
            HH3 = OUT['h3']
            h11, h12, h13 = torch.chunk(HH1, 3, dim=0)
            h21, h22, h23 = torch.chunk(HH2, 3, dim=0)
            h31, h32, h33 = torch.chunk(HH3, 3, dim=0)

            with torch.no_grad():
                sample = sample.cpu().numpy()
                h11 = h11.cpu().numpy()
                h12 = h12.cpu().numpy()
                h13 = h13.cpu().numpy()
                h21 = h21.cpu().numpy()
                h22 = h22.cpu().numpy()
                h23 = h23.cpu().numpy()
                h31 = h31.cpu().numpy()
                h32 = h32.cpu().numpy()
                h33 = h33.cpu().numpy()
                c = [sample, (h11, h12, h13), (h21, h22, h23), (h31, h32, h33)]
                reconstructed_x = pywt.waverec2(c, 'bior3.5', mode='reflect')

                out = torch.from_numpy(reconstructed_x)
            min_val = out.min()
            max_val = out.max()

            save_image(out, f"/root/autodl-tmp/image/img{i}.png", nrow=4, normalize=True,value_range=(-1, 1))




    model.eval()


    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path-val", type=str, default="/root/mamba/EUVP2")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=1)#原本预设的是256
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
