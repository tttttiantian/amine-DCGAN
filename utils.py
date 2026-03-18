import torch
import matplotlib.pyplot as plt
import torchvision

def save_generated_images(images, epoch, output_dir='outputs/images'):
    grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1,2,0))
    plt.title(f"Epoch {epoch}")
    plt.axis('off')
    plt.savefig(f"{output_dir}/epoch_{epoch}.png")
    plt.close()

def plot_loss(G_losses, D_losses):
    import matplotlib.pyplot as plt
    plt.plot(G_losses, label="G Loss")
    plt.plot(D_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
