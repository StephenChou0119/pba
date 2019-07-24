from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np

class CsvDataset(Dataset):
    '''
    root: 图片根目录
    csv_path csv存放路径

    '''
    def __init__(self, root, csv_path, transform=None,):
        super(CsvDataset, self).__init__()
        self.root = root
        self.target_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.target_frame)

    def __getitem__(self, item):
        img_name = self.target_frame.iloc[item, 0]
        target = int(self.target_frame.iloc[item, 1])
        img_fullname = os.path.join(self.root, img_name)
        img = Image.open(img_fullname)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
