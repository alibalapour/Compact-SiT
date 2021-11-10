from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image

from datasets.datasets_utils import getItem


# MHIST Histopathology Dataset
class MHIST(Dataset):
    def __init__(self, dataset_folder, transform=None, training_mode='finetune'):
        self.data = dataset_folder
        self.transform = transform
        self.training_mode = training_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = np.array(self.data[idx][0])
        X = Image.fromarray(X.astype(np.uint8))
        target = self.data[idx][1]
        return getItem(X, target, self.transform, self.training_mode)
