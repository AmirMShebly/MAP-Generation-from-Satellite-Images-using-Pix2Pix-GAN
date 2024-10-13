import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = os.listdir(self.root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = os.path.join(self.root, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_input(image=input_image)["image"]
        target_image = config.transform_output(image=target_image)["image"]

        return input_image, target_image