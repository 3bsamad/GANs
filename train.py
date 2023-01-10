import torch
from torchvision import transforms
import torch.nn as nn
from torch import optim
from cgan import Generator, Discriminator
from data import HandDataset


class Trainer:

    def __init__(self, root_dir, csv_file, batch_size, g_learning_rate, d_learning_rate,
                 transform=transforms.Compose([transforms.ToTensor()])):

        self.transform = transform
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.g_learning_rate = g_learning_rate
        self.d_learnng_rate = d_learning_rate
        hand_dataset = HandDataset(root_dir=root_dir, csv_file=csv_file, transform=self.transform)
        data_loader = torch.utils.data.DataLoader(hand_dataset, batch_size=batch_size, shuffle=True)

    def train(self):

        criterion = nn.BCELoss()
        g_optimizer = optim.Adam(Generator.parameters(), lr=self.g_learning_rate)
        d_optimizer = optim.Adam(Discriminator.parameters(), lr=self.d_learning_rate)

        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.data_loader, 0):
                # Update discriminator
                d_optimizer.zero_grad()
                real_data = data['image'].to(self.device)
                real_skintype = data['skinColor'].to(self.device)
                real_gender = data['gender'].to(self.device)

                d_real_output = Discriminator({'image': real_data, 'skinColor': real_skintype, 'gender': real_gender})
                d_real_error = criterion(d_real_output, torch.ones(d_real_output.size()).to(self.device))
                d_real_error.backward()

                noise = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
                fake_data = Generator({'noise': noise, 'skinColor': real_skintype, 'gender': real_gender})
                d_fake_output = Discriminator(
                    {'image': fake_data.detach(), 'skinColor': real_skintype, 'gender': real_gender})
                d_fake_error = criterion(d_fake_output, torch.zeros(d_fake_output.size()).to(self.device))
                d_fake_error.backward()

                d_optimizer.step()

                # Update generator
                g_optimizer.zero_grad()
                noise = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
                fake_data = Generator({'noise': noise, 'skinColor': real_skintype, 'skinColor': real_gender})
                d_fake_output = Discriminator({'image': fake_data, 'skinColor': real_skintype, 'gender': real_gender})
                g_error = -torch.mean(d_fake_output)
                g_error.backward()
                g_optimizer.step()

                if i % self.print_every == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f" % (
                    epoch, self.num_epochs, i, len(self.data_loader), d_real_error + d_fake_error, g_error))

        self.save_models()








def main():
    root_dir = '/home/bmw/Desktop/Projects/gans/GANs/dataset/Hands/Hands'
    csv_file = '/home/bmw/Desktop/Projects/gans/GANs/dataset/HandInfo.csv'

    Trainer_1 = Trainer(root_dir, csv_file, batch_size=16)
    Trainer_1.train()


if __name__ == '__main__':
    main()
