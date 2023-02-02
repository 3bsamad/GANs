import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch.utils.data as data
import numpy as np
from torchvision import transforms


class HandDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=True):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame['skinColor'] = pd.factorize(self.data_frame['skinColor'])[0]
        self.data_frame['gender'] = pd.factorize(self.data_frame['gender'])[0]
        self.data_frame['aspectOfHand'] = pd.factorize(self.data_frame['aspectOfHand'])[0]
        if self.transform:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 7])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        skintype = self.data_frame.iloc[idx, 3]
        gender = self.data_frame.iloc[idx, 2]
        aspectOfHand = self.data_frame.iloc[idx, 6]

        skin_color_onehot = torch.zeros(4)
        skin_color_onehot[skintype] = 1.0

        gender_onehot = torch.zeros(2)
        gender_onehot[gender] = 1.0

        aspect_of_hand_onehot = torch.zeros(4)
        aspect_of_hand_onehot[aspectOfHand] = 1.0

        label = torch.cat([skin_color_onehot, gender_onehot, aspect_of_hand_onehot])

        sample = {'image': image, 'label': label}
        return sample


class HandDataLoader:
    def __init__(self, dataset, batch_size=16, shuffle=True, split=0.8):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

        self.dataset_size = len(dataset)
        self.indices = list(range(self.dataset_size))
        self.split = int(np.floor(self.split * self.dataset_size))
        self.train_indices = self.indices[:self.split]
        self.val_indices = self.indices[self.split:]

    def get_train_loader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                           sampler=torch.utils.data.SubsetRandomSampler(self.train_indices))

    def get_val_loader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                           sampler=torch.utils.data.SubsetRandomSampler(self.val_indices))

    def get_test_loader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)


def main():
    root_dir = '/home/bmw/Desktop/Projects/gans/GANs/dataset/Hands/Hands'
    csv_file = '/home/bmw/Desktop/Projects/gans/GANs/dataset/HandInfo.csv'
    dataset = HandDataset(root_dir, csv_file)
    input = dataset[7600]
    print(input['image'].size())
    dataloader = HandDataLoader(dataset)
    trainloader = dataloader.get_train_loader()
    

# Perform some operation on the batch
# ...


if __name__ == "__main__":
    main()
