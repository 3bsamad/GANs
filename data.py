import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
from torchvision import transforms


class HandDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame['skinColor'] = pd.factorize(self.data_frame['skinColor'])[0]
        self.data_frame['gender'] = pd.factorize(self.data_frame['gender'])[0]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 7])
        image = Image.open(img_name)
        skintype = self.data_frame.iloc[idx, 3]
        gender = self.data_frame.iloc[idx, 2]
        transform = transforms.Compose([
            transforms.Resize((320, 240)),
            transforms.ToTensor()])
        image = transform(image)
        sample = {'image': image, 'skinColor': skintype, 'gender': gender}
        return sample



def main():
    root_dir = '/home/bmw/Desktop/Projects/gans/GANs/dataset/Hands/Hands'
    csv_file = '/home/bmw/Desktop/Projects/gans/GANs/dataset/HandInfo.csv'
    dataset = HandDataset(root_dir, csv_file)
    input = dataset[7600]
    print(input)

    condition = torch.zeros(input["skinColor"].shape[0], 2)
    condition[torch.arange(input["skinColor"].shape[0]), input["skinColor"]] = 1
    condition2 = torch.zeros(input["gender"].shape[0], 2)
    condition2[torch.arange(input["gender"].shape[0]), input["gender"]] = 1
    condition = torch.cat([condition, condition2], 1)
    x = torch.cat([input['image'].view(input['image'].size(0), -1), condition], 1)


if __name__ == "__main__":
    main()
