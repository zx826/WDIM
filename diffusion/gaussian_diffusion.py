import math
import numpy as np
import pywt
import torch
import torch as th
import torch.nn as nn
import enum
from diffusion.diffusion_utils import discretized_gaussian_log_likelihood, normal_kl
from loss import Hierarchical_SSIM,PerceptualLoss
from diffusers.models import AutoencoderKL
from timm.models.vision_transformer import  PatchEmbed,Mlp
import torch.nn.functional as F


def vae_decode(x):
    vae = AutoencoderKL.from_pretrained(f"sd-vae-ft-mse").to('cuda')
    img = vae.decode(x / 0.18215).sample
    return img

def mean_flat(tensor):  #取所有非批量维度的平均值。
    return tensor.mean(dim=list(range(1, len(tensor.shape)))) #求除了批次维度，从1~len(tensor.shape)的每一个维度上的平均值

def perloss(img1,img2):
    per_loss = PerceptualLoss().cuda()
    per_loss = per_loss(img1,img2)
    return per_loss
def HierarchicalSSIM(img1,img2):
    loss = Hierarchical_SSIM(img1, img2)
    return loss
class ModelMeanType(enum.Enum): #模型预测哪种类型的输出。

    PREVIOUS_X = enum.auto()  # 预测x_{t-1}的模型
    START_X = enum.auto()  # 预测x_0的模型
    EPSILON = enum.auto()  # 预测噪声 epsilon的模型


class ModelVarType(enum.Enum):# 用作模型的输出方差。添加了 LEARNED_RANGE 选项以允许模型进行预测FIXED_SMALL 和 FIXED_LARGE 之间的值，使其工作更容易。

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum): # 损失类型
    MSE = enum.auto()  # 使用原始 MSE 损失（学习方差时使用 KL）
    RESCALED_MSE = (enum.auto())  # 使用原始 MSE 损失（学习方差时使用 RESCALED_KL）
    KL = enum.auto()  # 使用变分下界
    RESCALED_KL = enum.auto()  # 与 KL 类似，但重新缩放以估计完整的 VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac): #warmup形式的beta
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)  # warmup_time作为分隔值总数（包括起始点和终止点），
                                                                                            # 即总共平均分成warmup_time个数
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    这是用于创建 Beta 计划的已弃用 API。请参阅 get_named_beta_schedule() 了解新的时间表库。
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
创建一个 beta 计划，离散化给定的 alpha_t_bar 函数，
     它定义了 (1-beta) 从 t = [0,1] 开始随时间的累积乘积。
     :param num_diffusion_timesteps: 生成的 beta 数量。
     :param alpha_bar: 一个 lambda，其参数 t 从 0 到 1，并且
                       产生 (1-beta) 的累积乘积
                       扩散过程的一部分。
     :param max_beta: 使用的最大 beta 值； 使用低于 1 的值
                      防止奇点.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    获取给定名称的预定义测试计划。Beta 时间表库由保持相似的 Beta 时间表组成
    在 num_diffusion_timesteps 的限制内。
    可以添加测试版时间表，但不应删除或更改，保持向后兼容性。
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


class to_MLP(nn.Module):
    def __init__(self, img_channels, img_height, img_width, hidden_dim, output_dim):
        super(to_MLP, self).__init__()
        input_dim = img_channels * img_height * img_width * 2  # 两个图像展平后的大小
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.img_channels = img_channels
        self.img_height = img_height
        self.img_width = img_width


    def forward(self, img1, img2):
        # 展平图像张量
        img1 = img1.view(img1.size(0), -1)
        img2 = img2.view(img2.size(0), -1)

        # 拼接两个图像张量
        x = th.cat((img1, img2), dim=1)

        # 前向传播
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(img1.size(0), self.img_channels, self.img_height, self.img_width)

        return x


# class TVLoss(nn.Module):
#     def __init__(self, TVLoss_weight=1):
#         super(TVLoss, self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self._tensor_size(x[:, :, 1:, :])
#         count_w = self._tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#
#     def _tensor_size(self, t):
#         return t.size()[1] * t.size()[2] * t.size()[3]




class GaussianDiffusion:     #用于训练和采样扩散模型的实用程序。
    def __init__(self,*,betas,model_mean_type,model_var_type,loss_type):#param betas：每个扩散时间步长的 beta 的一维 numpy 数组，从 T 开始，到 1

        # self.TVloss = TVLoss()
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)  # beta转换的numpy向量
        self.betas = betas
        assert len(betas.shape) == 1, "betas 必须 1-D"
        assert (betas > 0).all() and (betas <= 1).all()  #betas必须在0~1范围

        self.num_timesteps = int(betas.shape[0])  #时间步长也就是betas数量

        alphas = 1.0 - betas    #alphas定义
        self.alphas_cumprod = np.cumprod(alphas, axis=0)       #将alphas传入np.cumprod（计算累积乘积量的函数）中，这是alphas(t)_bar
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])#alphas（t）_bar从第0项开始传进来（不包括最后一项）,第0项用1填充，这是alphas(t-1)_bar
                                                                              #np.append（）是两个数组的拼接函数
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)#alphas（t）_bar从第1项开始传进来（去除第0项），用0做最后一项，这是alphas（t+1）_bar
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # 扩散 q(x_t | x_{t-1}) 及其他计算
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)        # 根号alphas(t)_bar
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod) # 根号1—alphas(t)_bar
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)  # log(1.0 - alphas(t)_bar)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)    # 根号1.0 / alphas(t)_bar
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1) # sqrt(1.0 / alphas(t)_bar  - 1)

        # 计算后验 q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (              #真实后验方差
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 由于扩散链开始时的后验方差为 0。裁剪对数计算
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([]) ##对上面的方差取个对数，并对其截断，防止第0项为0，用posterior_variance[1]代替

        self.posterior_mean_coef1 = (       #后验均值的第一个系数   在论文公式10
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (        #后验均值的第二个系数   在论文公式10
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )


    '''q分布是真实分布，区别于神经网络预测的p'''
    def q_mean_variance(self, x_start, t):#获取分布 q(x_t | x_0)。x_start：无噪声输入的 [N x C x ...] 张量。param t: 扩散步数（负 1）。 这里，0表示一步。
                                           #:return: 一个元组（均值、方差、log_variance），所有 x_start 的形状。
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start  #均值
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)      #方差
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape) #log(1.0 - alphas(t)_bar)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):#加噪，x_start: 初始数据批次，t: 扩散步数（负 1）。 这里，0表示一步。return: x_start 的嘈杂版本。
        if noise is None:
            noise = th.randn_like(x_start)    # 噪声
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):# 计算扩散后验的均值和方差：q(x_{t-1} | x_t, x_0)
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )   #后验均值
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)#后验方差
        posterior_log_variance_clipped = _extract_into_tensor(  #后验log方差
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    '''神经网络的预测'''
    def p_mean_variance(self, model, x, t, y,H1,H2,H3, clip_denoised=True, denoised_fn=None, model_kwargs=None):#应用模型得到 p(x_{t-1} | x_t)，以及预测初始x，x_0。
                                                                  #origin: 模型，它需要一个信号和一批时间步长作为输入
                                                                  #x: 时间t时的[N x C x...] 张量。
                                                                  #denoised_fn: 如果不为None，一个在用于采样之前应用于x_start预测的函数。在clip_denoised之前应用。

        if model_kwargs is None: #model_kwargs: 如果不为None，一个包含额外关键字参数的字典，用于传递给模型。这可以用于条件设置。
            model_kwargs = {}

        B, C = x.shape[:2]    #[:2]表示取x前两个维度的大小。因此，B和C分别表示张量x的第一个和第二个维度的大小
        assert t.shape == (B,)
        model_output,h1,h2,h3 = model(x, t, y,H1,H2,H3, **model_kwargs) #将x和t放入模型里，**kwargs将不定长度的 【键值对 key-value 】作为参数传递给一个函数，即在一个函数里传入带名字的参数

        if isinstance(model_output, tuple):  #元组（tuple）是 Python 中的一种数据结构，类似于列表，但是元组是不可变的（immutable）序列
            model_output, extra = model_output
        else:
            extra = None
        '''判断生成的是什么类型的输出'''
         # 如果是可学习的方差
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:# ModelVarType.LEARNED直接预测方差
                                                                                      # #ModelVarType.LEARNED_RANGE预测方差的范围，论文提出的改进点公式14
            #print(model_output.shape,x.shape[:2])
            assert model_output.shape == (B, C * 3, *x.shape[2:])
            model_output, model_var_values, model_zero = th.split(model_output, C, dim=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape) #后验方差
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)  #beta方差
            # The model_var_values is [-1, 1] for [min_var, max_var]
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)  #以上是论文公式14
        else:  #如果不学习方差

            model_variance, model_log_variance = {
                # 对于fixedlarge，我们设置初始（对数）方差，如下所示。以获得更好的解码器对数似然。
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        '''对x进行一定的处理'''
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output) #通过学习到的噪声来得到x0
            )
        #pred_xstart = 0.9*pred_xstart + 0.1*model_zero
        # model = to_MLP(img_channels=model_zero.shape[1],img_height=model_zero.shape[2], img_width=model_zero.shape[3], hidden_dim=1200, output_dim=model_zero.shape[1]*model_zero.shape[2]*model_zero.shape[3])
        # model = model.to('cuda')
        # xstart = model(model_zero,pred_xstart)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start = pred_xstart, x_t = x, t = t)  #计算扩散后验的均值：q(x_{t-1} | x_t, x_0)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {"mean": model_mean,"model_zero": model_zero,"variance": model_variance,"log_variance": model_log_variance,"pred_xstart": pred_xstart,"extra": extra,"noise":model_output,"h1":h1,"h2":h2,"h3":h3 }

    def _predict_xstart_from_eps(self, x_t, t, eps): #从噪声中预测x0
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart): #从x0预测噪声
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
         给定函数 cond_fn ，计算上一步的平均值
         计算条件对数概率的梯度X
         特别是，通过cond_fn 计算 grad(log(p(y|x)))，我们想要y的条件。
         这使用了 Sohl-Dickstein 等人的调节策略。 （2015）。
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        计算 p_mean_variance 输出是什么，如果模型的得分函数以 cond_fn 为条件。
        关于 cond_fn 的详细信息，请参阅 condition_mean()。
        与condition_mean()不同，它使用条件策略
         来自 Song 等人 (2020)。
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"]) #通过x0预测噪声eps
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(x_start=out["pred_xstart"], x_t=x, t=t)
        return out

    def p_sample(self,model,x,t,y,H1, H2, H3,clip_denoised=True,denoised_fn=None,cond_fn=None, model_kwargs=None):#在给定的时间步长从模型中采样 x_{t-1}。
        """
         :origin: 采样的模型。
         :x: x_{t-1} 处的当前张量。
         :t：t 的值，第一个扩散步骤从 0 开始。
         :Clip_enoished: 如果为 True，则将 x_start 预测剪辑为 [-1, 1]。
         :denoized_fn: 如果不是 None，则适用于
             x_start 在用于采样之前进行预测。
         :cond_fn: 如果不是 None，这是一个起作用的梯度函数与模型类似。
         :model_kwargs: 如果不是 None，则为额外关键字参数的字典传递给模型。 这可以用于调节。
         :return: 包含以下键的字典：-“样本”：模型中的随机样本。
                                - 'pred_xstart'：x_0 的预测。
        """

        out = self.p_mean_variance(model,x,t,y,H1,H2,H3,clip_denoised=clip_denoised,denoised_fn=denoised_fn,model_kwargs=model_kwargs)
                                #输出out{"mean": model_mean,"variance": model_variance,"log_variance": model_log_variance,"pred_xstart": pred_xstart,"extra": extra}
        noise = th.randn_like(x)  # 随机生成噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # 当 t == 0时没有噪声
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise #均值+方差*噪声  作为t-1步的sample
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self,model,shape,noise=None,y=None, H1=None, H2=None, H3=None, clip_denoised=True, denoised_fn=None,cond_fn=None,model_kwargs=None,device=None,progress=False):
        """
         从模型生成样本。
         :origin：模型模块。
         :shape：样本的形状（N、C、H、W）。
         :noise：如果指定，则为来自编码器采样的噪声。应该与“形状”具有相同的形状。
         :Clip_enoished: 如果为 True，则将 x_start 预测剪辑为 [-1, 1]。
         :denoized_fn: 如果不是 None，则适用于x_start 在用于采样之前进行预测。
         :cond_fn: 如果不是 None，这是一个起作用的梯度函数与模型类似。
         :model_kwargs: 如果不是 None，则为额外关键字参数的字典传递给模型。 这可以用于调节。
         :device：如果指定，则为创建样本的设备。如果未指定，则使用模型参数的设备。
         :Progress: 如果为 True，则显示 tqdm 进度条。
         :return: 一批不可微分的样本。
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            y=y,
            H1=H1,
            H2=H2,
            H3=H3,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample

        t = list(range(self.num_timesteps))[::-1]
        t = th.tensor([t[0]] * shape[0], device=device)
        out = self.p_mean_variance(model, noise, t , y, H1, H2, H3, clip_denoised=clip_denoised, denoised_fn=denoised_fn,
                                   model_kwargs=model_kwargs)


        return final["sample"],out

    def p_sample_loop_progressive(self,model,shape,noise=None,y=None,H1=None, H2=None, H3=None,clip_denoised=True,denoised_fn=None,cond_fn=None,model_kwargs=None,device=None,progress=False):
        """
          从模型生成样本并产生中间样本扩散的每个时间步长。
          参数与 p_sample_loop() 相同。返回字典上的生成器，其中每个字典都是以下内容的返回值p_sample()。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise  #给定的噪声
        else:
            img = th.randn(*shape, device=device) #随机生成的噪声
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # 延迟导入，以便我们不依赖 tqdm。
            from tqdm.auto import tqdm
            indices = tqdm(indices)


        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():

                out = self.p_sample(model,img,t,y, H1, H2, H3,clip_denoised=clip_denoised,denoised_fn=denoised_fn,cond_fn=cond_fn,model_kwargs=model_kwargs)
                                                         #out{"sample": sample, "pred_xstart": out["pred_xstart"]}
                yield out
                img = out["sample"]#sample


    def ddim_sample(self,model,x,t,clip_denoised=True,denoised_fn=None,cond_fn=None,model_kwargs=None,eta=0.0):
        """
        从使用 DDIM模型采样 x_{t-1} .
        和 p_sample()相同的用法.
        """
        out = self.p_mean_variance(model,x,t,clip_denoised=clip_denoised,denoised_fn=denoised_fn,model_kwargs=model_kwargs)
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # 通常我们的模型输出 epsilon，但我们重新推导它如果我们使用 x_start 或 x_prev 预测。
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"]) #从x0中预测噪声

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)         #alphas(t)_bar
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)  #alphas(t-1)_bar
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(self,model,x,t,clip_denoised=True,denoised_fn=None,cond_fn=None,model_kwargs=None,eta=0.0):
        """
        Sample x_{t+1} from the origin using DDIM reverse ODE.
        """
        assert eta == 0.0, "逆常微分方程仅适用于确定性路径"
        out = self.p_mean_variance(model,x,t,clip_denoised=clip_denoised,denoised_fn=denoised_fn,model_kwargs=model_kwargs)
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our origin outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * th.sqrt(alpha_bar_next) + th.sqrt(1 - alpha_bar_next) * eps #这不是xt的公式吗

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the origin using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        使用 DDIM 从模型中采样并生成中间样本
        DDIM 的每个时间步长。
        与 p_sample_loop_progressive() 的用法相同。
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, y, H1, H2, H3, clip_denoised=True, model_kwargs=None
    ):
        """
        获取变分下界的术语。结果单位是位（而不是人们所期望的 nat）。这可以与其他论文进行比较。
         :return: 具有以下键的字典：
                  -“输出”：NLL 或 KL 的形状 [N] 张量。
                  - 'pred_xstart'：x_0 预测。
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(       #真实均值和方差
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(   #预测out{"mean": model_mean,"variance": model_variance,"log_variance": model_log_variance,"pred_xstart": pred_xstart,"extra": extra}
            model, x_t, t, y, H1, H2, H3, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(                      #kl散度？
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # 在第一个时间步返回解码器 NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, y, H1, H2, H3, model_kwargs=None, noise=None):
        """
        计算单个时间步的训练损失。
         ：origin：评估损失的模型。
         :x_start: 输入的 [N x C x ...] 张量。
         : t: 一批时间步长索引。
         :model_kwargs: 如果不是 None，则为额外关键字参数的字典
             传递给模型。 这可以用于调节。
         ：noise：如果指定，则尝试消除的特定高斯噪声。
         : 一个字典，其键为“loss”，包含形状为 [N] 的张量。一些均值或方差设置也可能有其他键。
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)


        noise1 = th.randn_like(x_start)
        # x_t_1 = self.q_sample(x_start, t - 1, noise=noise1)
        # noise2 = th.randn_like(x_start)
        # x_t_2 = self.q_sample(x_start, t - 2, noise=noise2)



        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                H1=H1,
                H2=H2,
                H3=H3,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output,HL,HH,LH = model(x_t, t,y, H1, H2, H3, **model_kwargs)



            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                #print(model_output.shape,x_t.shape[:2], *x_t.shape[2:])
                assert model_output.shape == (B, C * 3, *x_t.shape[2:])
                #model_output, time = th.split(model_output, C * 2, dim=1)
                model_output, model_var_values,model_zero = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values, model_zero.detach()], dim=1),HL,HH,LH
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,  #因此，这个 lambda 函数无论传递什么参数，返回的始终是 frozen_out
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    y=y,
                    H1=H1,
                    H2=H2,
                    H3=H3,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # out = self.p_mean_variance(model, x_t, t, y, H1, H2, H3, clip_denoised=True, model_kwargs=model_kwargs)
            # # out1 = self.p_mean_variance(model, x_t_1, t, y, clip_denoised=True, model_kwargs=model_kwargs)
            # nonzero_mask = (
            #     (t != 0).float().view(-1, *([1] * (len(x_start.shape) - 1)))
            # )  # 当 t == 0时没有噪声
            # sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise  # 均值+方差*噪声  作为t-1步的sample
            # sample1 = out1["mean"] + nonzero_mask * th.exp(0.5 * out1["log_variance"]) * noise1  # 均值+方差*噪声  作为t-2步的sample

            h11, h12, h13 = th.chunk(HL, 3, dim=0)
            h21, h22, h23 = th.chunk(HH, 3, dim=0)
            h31, h32, h33 = th.chunk(LH, 3, dim=0)
            H11, H12, H13 = th.chunk(H1, 3, dim=0)
            H21, H22, H23 = th.chunk(H2, 3, dim=0)
            H31, H32, H33 = th.chunk(H3, 3, dim=0)

            a = torch.cat([h11, h12, h13], dim=1)
            b = torch.cat([h21, h22, h23], dim=1)
            c = torch.cat([h31, h32, h33], dim=1)
            A = torch.cat([H11, H12, H13], dim=1)
            B = torch.cat([H21, H22, H23], dim=1)
            C = torch.cat([H31, H32, H33], dim=1)

            # with th.no_grad():
            #     sample = sample.cpu().numpy()
            #     X_t_1 = x_t_1.cpu().numpy()
            #     h11 = h11.cpu().numpy()
            #     h12 = h12.cpu().numpy()
            #     h13 = h13.cpu().numpy()
            #     h21 = h21.cpu().numpy()
            #     h22 = h22.cpu().numpy()
            #     h23 = h23.cpu().numpy()
            #     h31 = h31.cpu().numpy()
            #     h32 = h32.cpu().numpy()
            #     h33 = h33.cpu().numpy()
            #     H11 = H11.cpu().numpy()
            #     H12 = H12.cpu().numpy()
            #     H13 = H13.cpu().numpy()
            #     H21 = H21.cpu().numpy()
            #     H22 = H22.cpu().numpy()
            #     H23 = H23.cpu().numpy()
            #     H31 = H31.cpu().numpy()
            #     H32 = H32.cpu().numpy()
            #     H33 = H33.cpu().numpy()
            #
            #     x1 = pywt.idwt2((sample, (h11,h12,h13)),'haar')
            #     x2 = pywt.idwt2((x1, (h21, h22, h23)), 'haar')
            #     x3 = pywt.idwt2((x2, (h31, h32, h33)), 'haar')
            #     x3 = th.from_numpy(x3).to('cuda')
            #     X1 = pywt.idwt2((X_t_1, (H11, H12, H13)), 'haar')
            #     X2 = pywt.idwt2((X1, (H21, H22, H23)), 'haar')
            #     X3 = pywt.idwt2((X2, (H31, H32, H33)), 'haar')
            #     X3 = th.from_numpy(X3).to('cuda')
            #     sample = th.from_numpy(sample).to('cuda')
            #     X_t_1 = th.from_numpy(X_t_1).to('cuda')
            #     x1 = th.from_numpy(x1).to('cuda')
            #     x2 = th.from_numpy(x2).to('cuda')

            terms["l1"] = mean_flat(th.abs(A - a)) + mean_flat(th.abs(B - b)) + mean_flat(th.abs(C - c))
            # terms["l2"] = mean_flat((sample - X_t_1)**2) + mean_flat((x1 - th.from_numpy(X1).to('cuda'))**2) + mean_flat((x2 - th.from_numpy(X2).to('cuda'))**2) + mean_flat((x3 - X3)**2)
            # terms["l2"] = mean_flat((sample - x_t_1) ** 2)
            # terms["l1"] = 0



            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t)[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape == model_zero.shape
            #print(model_output.shape ,target.shape)
            terms["mse"] = mean_flat((target - model_output) ** 2)
            #x0 = self._predict_xstart_from_eps(x_t, t, model_output)
            #print(_extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape).shape,model_zero.shape)

            terms["l2"] = mean_flat(th.abs(target - model_output))
            #x_p0 = self._predict_xstart_from_eps(x_t, t, model_output)
            # x_pt = self.q_sample(x_start, t, noise=model_output)
            # img1 = vae_decode(x_pt)
            # img2 = vae_decode(x_t)
            terms["Hierarchical_SSIM"] = 0

            # terms["Hierarchical_SSIM"] = self.TVloss(sample) + self.TVloss(x1)+ self.TVloss(x2)

            #print(terms["mse"],terms["Hierarchical_SSIM"])
            # img1 = vae_decode(model_zero)
            # img2 = vae_decode(y)
            # terms["per_loss"] = perloss(img1, img2)
            # terms["Hierarchical_SSIM"] = HierarchicalSSIM(img1, img2)
            terms["per_loss"] = 0
            #terms["Hierarchical_SSIM"] = 0

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]+terms["l1"]+terms["l2"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms ,terms["Hierarchical_SSIM"], terms["per_loss"]

    def _prior_bpd(self, x_start):
        """
        获取变分下限的先验 KL 项，测量单位为每暗位数。
         该术语无法优化，因为它仅取决于编码器。
         :param x_start: 输入的 [N x C x ...] 张量。
         :return: 一批 [N] KL 值（以位为单位），每批元素一个。
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        计算整个变分下界，以每暗位数测量，
         以及其他相关数量。
         ：param origin：评估损失的模型。
         :param x_start: 输入的 [N x C x ...] 张量。
         :param Clip_enoished: 如果为 True，则剪辑去噪样本。
         :param model_kwargs: 如果不是 None，则为额外关键字参数的字典
             传递给模型。 这可以用于调节。
         :return: 包含以下键的字典：
                  -total_bpd：每批元素的总变分下限。
                  -prior_bpd：下限中的先验项。
                  - vb：下界项的 [N x T] 张量。
                  - xstart_mse：每个时间步的 x_0 MSE 的 [N x T] 张量。
                  - mse：每个时间步的 epsilon MSE 的 [N x T] 张量。
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape): #接收三个参数：arr为要转换的 numpy 数组，timesteps为时间步，broadcast_shape为广播形状。
                                                           #return: 形状为 [batch_size, 1, ...] 的张量，其中形状有 K 个暗淡。
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()  #将 numpy数组转换为 PyTorch 张量，并根据timesteps选择相应的时间步
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]        # 如果res的维度少于broadcast_shape的维度，则进行以下循环操作，将res的维度扩展到与broadcast_shape相同。
                                    #在res的最后一个维度上添加一个维度，实现维度扩展。
    return res + th.zeros(broadcast_shape, device=timesteps.device) #将res与一个全零张量相加，使其形状与broadcast_shape相同
