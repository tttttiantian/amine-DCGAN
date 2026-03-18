import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.generator import Generator, weights_init
from models.discriminator import Discriminator
from dataset.dataset import AnimeDataset
from utils import save_generated_images, plot_loss
import os

# =====================
# 参数
# =====================
z_dim = 100
lr = 0.0002
batch_size = 64
epochs = 100
data_dir = '/kaggle/input/datasets/soumikrakshit/anime-faces/data'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 数据集
# =====================
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
dataset = AnimeDataset(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# =====================
# 模型
# =====================
G = Generator(z_dim).to(device)
D = Discriminator().to(device)
G.apply(weights_init)
D.apply(weights_init)

optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

G_losses = []
D_losses = []

# =====================
# 训练循环
# =====================
for epoch in range(epochs):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        real_labels = torch.ones(batch_size, 1).to(device) * 0.9
        fake_labels = torch.zeros(batch_size, 1).to(device)

        optimizer_D.zero_grad()
        real_imgs_noisy = real_imgs + 0.05*torch.randn_like(real_imgs)
        d_loss_real = criterion(D(real_imgs_noisy), real_labels)

        z = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_imgs = G(z)
        d_loss_fake = criterion(D(fake_imgs.detach()), fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()

        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] Step [{i}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # 每10轮保存生成图片
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(64, z_dim, 1, 1).to(device)
            fake = G(z).cpu()
            save_generated_images(fake, epoch+1)

# =====================
# 保存模型
# =====================
os.makedirs('outputs/models', exist_ok=True)
torch.save(G.state_dict(), 'outputs/models/generator.pth')
torch.save(D.state_dict(), 'outputs/models/discriminator.pth')

# =====================
# 绘制loss
# =====================
plot_loss(G_losses, D_losses)
