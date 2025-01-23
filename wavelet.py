import torch
import torch.nn as nn
import pywt



def dwt_init(x,k):
    x0 = x[:, 0, :, :].cpu()
    x1 = x[:, 1, :, :].cpu()
    x2 = x[:, 2, :, :].cpu()
    wave0 = pywt.wavedec2(x0, 'bior3.5', mode='periodization', level=k) #mode='periodization'周期  ； mode='reflect'反射   ；'db4'；'bior3.5'；'coif5'；'haar'
    wave1 = pywt.wavedec2(x1, 'bior3.5', mode='periodization', level=k)
    wave2 = pywt.wavedec2(x2, 'bior3.5', mode='periodization', level=k) # k次小波变换

    '''K次小波低频分量'''
    LL1 = torch.from_numpy(wave0[0]).cuda()
    LL2 = torch.from_numpy(wave1[0]).cuda()
    LL3 = torch.from_numpy(wave2[0]).cuda() # 转为tensor
    x_LL = torch.stack((LL1,LL2,LL3),dim=1).cuda()
    '''K次小波高频分量'''
    high = []
    for i in range(1, k+1):
        (HL1, LH1, HH1), (HL2, LH2, HH2), (HL3, LH3, HH3) = wave0[i], wave1[i], wave2[i]  #三个通道分别分解出的高频分量

        HL1, HL2, HL3 = torch.from_numpy(HL1).cuda(), torch.from_numpy(HL2).cuda(), torch.from_numpy(
            HL3).cuda()  # 转为tensor向量
        x_HL = torch.stack((HL1, HL2, HL3), dim=1).cuda()

        LH1, LH2, LH3 = torch.from_numpy(LH1).cuda(), torch.from_numpy(LH2).cuda(), torch.from_numpy(
            LH3).cuda()  # 转为tensor向量
        x_LH = torch.stack((LH1, LH2, LH3), dim=1).cuda()

        HH1, HH2, HH3 = torch.from_numpy(HH1).cuda(), torch.from_numpy(HH2).cuda(), torch.from_numpy(
            HH3).cuda()  # 转为tensor向量
        x_HH = torch.stack((HH1, HH2, HH3), dim=1).cuda()
        h = torch.cat((x_HL, x_LH, x_HH), 0).cuda()
        high.append(h)  #不同次分解的高频集合
    return x_LL, high, k


# shape = (3, 3, 4, 5)  # 3个样本，每个样本3个通道，高度为4，宽度为5
# x = torch.randn(shape)
# x_LL,high,k = dwt_init(x)
# print(high[k-1])

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x,k):
        return dwt_init(x,k)

# from torchvision.utils import save_image
# import torch
# from torchvision import transforms
# from PIL import Image
# # 定义转换，包括调整大小和转换为Tensor
# transform = transforms.Compose([
#     transforms.Resize((256, 256)),  # 调整大小
#     transforms.ToTensor(),  # 转换为Tensor，并归一化到 [0, 1]
# ])
# # 读取图像并应用转换
# image_path = r"D:\A_zhangxin\diffusion-mamba1\adata\val1\0\6670.png"  # 替换为你的图片路径
# image = Image.open(image_path).convert('RGB')  # 读取并转换为RGB格式
# image_tensor = transform(image)  # 应用转换
#
# # 添加批次维度
# image_tensor = image_tensor.unsqueeze(0)  # 形状变为 (1, c, h, w)
#
# dwt = DWT()
# ll,h,_ = dwt(image_tensor,3)
# t1,t2,t3 = torch.chunk(h[2],3,dim=0)
# min_val = t3.min()
# max_val = t3.max()
# print(min_val,max_val,h[1].shape)
# save_image(t3, f"D:\A_zhangxin\diffusion-mamba1\generate\\h2_3.png", nrow=1, normalize=True,value_range=(-0.3, 0.3))

