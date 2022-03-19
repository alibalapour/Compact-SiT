from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image

from dataset.datasets_utils import getItem


# Custom Histopathology
class CustomDataset(Dataset):
    def __init__(self, dataset_folder, transform=None, training_mode='finetune', return_name=False):
        self.data = dataset_folder
        self.transform = transform
        self.training_mode = training_mode
        self.return_name = return_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = np.array(self.data[idx][0])
        X = Image.fromarray(X.astype(np.uint8))
        target = self.data[idx][1]
        print(self.return_name, self.data.imgs[idx])
        if self.return_name:
            return getItem(X, target, self.transform, self.training_mode), self.data.imgs[idx]
        return getItem(X, target, self.transform, self.training_mode)
