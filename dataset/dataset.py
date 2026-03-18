import os
from PIL import Image
from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = [f for f in os.listdir(root) if f.endswith(('jpg','png','jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.files[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
