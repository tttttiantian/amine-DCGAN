import torch
from models.generator import Generator
from utils import save_generated_images
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
z_dim = 100
G = Generator(z_dim).to(device)
G.load_state_dict(torch.load("outputs/models/generator.pth", map_location=device))
G.eval()

os.makedirs('outputs/images', exist_ok=True)
with torch.no_grad():
    z = torch.randn(64, z_dim, 1, 1).to(device)
    fake_images = G(z).cpu()
    save_generated_images(fake_images, "demo")
print("Generated images saved to outputs/images/demo.png")
