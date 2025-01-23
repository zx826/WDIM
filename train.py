import torch
import pywt
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.utils.data import SequentialSampler
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from glob import glob
from time import time
import argparse
import logging
import os
from Model import Mamba, MambaConfig
from diffusion import create_diffusion
from wavelet import DWT
from diffusion import gaussian_diffusion as gd





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


def load_checkpoint(checkpoint_path, model, opt, args):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        # 加载模型状态
        model.load_state_dict(checkpoint["model"])
        # 加载优化器状态
        opt.load_state_dict(checkpoint["opt"])
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
    config = MambaConfig(d_model=1280, n_layers=28)  # 模型设置d_model=1152, n_layers=28
    model = Mamba(config)
    #ema = deepcopy(model).to('cuda')  # 创建模型的 EMA以在训练后使用  deepcopy将某一个变量的值赋值给另一个变量(两个变量地址不同)
    #requires_grad(ema, False)  # 不需要梯度
    model = model.to('cuda')
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    dwt = DWT()
    #vae = AutoencoderKL.from_pretrained(f"sd-vae-ft-{args.vae}").to('cuda')  # 通过model.from_pretrained函数来加载预训练模型
    logger.info(f"WDIM Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 设置优化器（使用默认的 Adam betas=（0.9， 0.999） 和 1e-5 的恒定学习率）:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)

    # 设置数据:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    data_raw = ImageFolder(args.data_path, transform=transform)  # ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名
    data_gt = ImageFolder(args.data_path_gt, transform=transform)
    raw = SequentialSampler(data_raw)
    gt = SequentialSampler(data_gt)
    # combined_dataset = ConcatDataset([raw, gt])
    loader_raw = DataLoader(  # 深度学习训练的流程： 1. 创建Dateset 2. Dataset传递给DataLoader 3. DataLoader迭代产生训练数据提供给模型
        # Dataset负责建立索引到样本的映射，DataLoader负责以特定的方式从数据集中迭代的产生 一个个batch的样本集合
        data_raw,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=raw,
        pin_memory=True,
        drop_last=True
    )
    loader_gt = DataLoader(
        data_gt,
        batch_size=int(args.global_batch_size),
        shuffle=False,
        sampler=gt,
        pin_memory=True,
        drop_last=True
    )

    val_dataset = ImageFolder(args.data_path_val, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(1),
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )
    logger.info(f"raw_Data contains {len(data_raw):,} images ({args.data_path})")

    # 为训练准备模型:
    model.train()  # origin.train()的作用是启用 Batch Normalization 和 Dropout。。  重要！这样就可以嵌入 dropout 以实现无分类器引导

    # 用于监视/日志记录目的的变量:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    loss_loss1 = 0
    SSIM_loss = 0
    per_loss = 0
    m_ = 0

    logger.info(f"Training for {args.epochs} epochs...")
    writer = SummaryWriter(log_dir='/root/tf-logs')
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
        success = load_checkpoint(checkpoint_path,model, opt, args)
    else:
        success = False
    if success:
        print("Checkpoint loaded successfully. Continuing training...")
    else:
        print("No checkpoint found. Starting training from scratch...")

    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")

        for k, ((y, _), (x, _)) in enumerate(zip(loader_raw, loader_gt)):
            start_time = time()
            x = x.to('cuda')
            y = y.to('cuda')
            x,_,_ = dwt(x,3)
            y,h,_ = dwt(y,3)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device='cuda')  # 随机生成时刻t
            loss_dict, SSIM, PERLOSS = diffusion.training_losses(model, x, t, y, h[0], h[1], h[2],model_kwargs=None)
            loss1 = loss_dict["loss"].mean()  # 损失函数均值
            loss = loss1 + SSIM + PERLOSS
            opt.zero_grad()
            loss.backward()
            opt.step()

            end_time = time()
            m = (end_time - start_time)

            # 记录损失值:
            running_loss += loss.item()
            loss_loss1 += loss1.item()
            SSIM_loss += SSIM
            per_loss += PERLOSS
            m_ += m
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # 明确训练速度:
                torch.cuda.synchronize()  # PyTorch的 torch.cuda.synchronize () 函数来同步GPU上的所有操作
                # steps_per_sec = log_steps / (end_time - start_time)

                # 减少所有流程的损失历史:
                m_ = torch.tensor(m_ / log_steps, device='cuda').clone().detach()
                avg_loss = torch.tensor(running_loss / log_steps, device='cuda')
                avg_loss1 = torch.tensor(loss_loss1 / log_steps, device='cuda')
                avg_SSIM = torch.tensor(SSIM_loss / log_steps, device='cuda')
                avg_PERLOSS = torch.tensor(per_loss / log_steps, device='cuda')
                print(f"'epoch': {epoch}  (step={train_steps:07d}) Train Loss: {avg_loss:.4f},LOSS:{avg_loss1:.4f},SSIM: {avg_SSIM:.4f} ,PERLOSS:{avg_PERLOSS:.4f},log_steps:{log_steps:.4f},time:{m_:.4f}")
                writer.add_scalar('avg_loss', avg_loss, train_steps)
                # 重置监控变量:
                running_loss = 0
                loss_loss1 = 0
                SSIM_loss = 0
                per_loss = 0
                log_steps = 0
            # 保存checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/model.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        '''验证'''
        if (epoch+1) % 2 == 0 and train_steps > 0:
            i=0
            for u, _ in val_loader:
                i = i + 1
                u = u.to('cuda')
                with torch.no_grad():
                    # z, _, _ = dwt(z, 3)
                    u, H, _ = dwt(u, 3)
                    noise = torch.randn_like(u).to('cuda')

                    sample, OUT = diffusion.p_sample_loop(
                        model, noise.shape, noise, u, H[0], H[1], H[2], clip_denoised=False, model_kwargs=None,progress=True, device='cuda')
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
                        c = [sample, (h11, h12, h13),(h21, h22, h23), (h31, h32, h33)]
                        reconstructed_x = pywt.waverec2(c,  'bior3.5', mode='periodization')

                        out = torch.from_numpy(reconstructed_x)
                    # min_val = out.min()
                    # max_val = out.max()
                    # print(min_val,max_val)

                    # save_image(out, f"/root/autodl-tmp/generate/img{i}.png", nrow=4, normalize=True,value_range=(min_val, max_val))
                    save_image(out, f"/root/autodl-tmp/image{epoch+1}/img{i}.png", nrow=4, normalize=True, value_range=(-1, 1))





    model.eval()

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/root/mamba/adata/train/raw")
    parser.add_argument("--data-path-gt", type=str, default="/root/mamba/adata/train/gt")
    parser.add_argument("--data-path-val", type=str, default="/root/mamba/adata/val")
    parser.add_argument("--results-dir", type=str, default="/root/autodl-tmp/results")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=4)#原本预设的是256
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")  # Choice doesn't affect training
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
