import torch
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F
from data import HandDataset

num_epochs = 5


class Generator(nn.Module):
    def __init__(self, latent_dim=100, skinColor_dim=3, gender_dim=2):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.skintype_dim = skinColor_dim
        self.gender_dim = gender_dim
        self.input_size = latent_dim + skinColor_dim + gender_dim
        self.hidden_size = 256
        self.output_size = 76800
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self, input, skintype, gender):
        # one-hot encode the skintype and gender
        skintype = F.one_hot(skintype, num_classes=4)  # very fair, fair, medium, dark
        gender = F.one_hot(gender, num_classes=2)  # male, female
        # concatenate the one hot encoded vectors
        condition = torch.cat([skintype, gender], 1)
        x = torch.cat([input.view(input.size(0), -1), condition], 1)
        x = F.dropout(F.leaky_relu(self.fc1(x),0.2))
        x = F.dropout(F.leaky_relu(self.fc2(x),0.2))
        x = torch.sigmoid(self.fc3(x))
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input_size = 76800 + 5  # 320x240 images + 6-dimensional condition vector
        self.hidden_size = 256
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = torch.cat([input['image'].view(input['image'].size(0), -1), input['condition']], 1)
        x = F.dropout(F.leaky_relu(self.fc1(x),0.2))
        x = F.dropout(F.leaky_relu(self.fc2(x),0.2))
        x = self.sigmoid(self.fc3(x))
        return x




class CGAN:
    def __init__(self, root_dir, csv_file, batch_size, g_learning_rate, d_learning_rate, transform=transforms.Compose([transforms.ToTensor()])):
        
        self.transform = transform
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.g_learning_rate = g_learning_rate
        self.d_learnng_rate = d_learning_rate
        
        self.hand_dataset = HandDataset(root_dir=root_dir, csv_file=csv_file, transform=self.transform)
        self.data_loader = torch.utils.data.DataLoader(self.hand_dataset, batch_size=batch_size, shuffle=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        self.g_loss = nn.BCELoss()
        self.d_loss = nn.BCELoss()
        
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=g_learning_rate)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=d_learning_rate)

        # Create generator and discriminator
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for i, data in enumerate(self.data_loader, 0):
                image = data['image'].to(self.device)
                skintype = data['skinColor'].to(self.device)
                gender = data['gender'].to(self.device)

                # Get the one-hot encoded vectors
                skintype_ohe = F.one_hot(skintype, num_classes=4)
                gender_ohe = F.one_hot(gender, num_classes=2)
                # Concatenate the one hot encoded vectors
                condition = torch.cat([skintype_ohe, gender_ohe], 1)
                
                # Generate fake images
                fake_image = self.generator(image, skintype, gender)
                
                # Get the real and fake labels
                real_label = torch.ones(self.batch_size, 1).to(self.device)
                fake_label = torch.zeros(self.batch_size, 1).to(self.device)
                
                # Train the discriminator on real images
                d_real = self.discriminator({"image": image, "condition":condition})
                d_real_loss = d_loss(d_real, real_label)
                
                # Train the discriminator on fake images
                d_fake = self.discriminator({"image": fake_image.detach(), "condition":condition})
                d_fake_loss = d_loss(d_fake, fake_label)
                
                # Compute the total discriminator loss
                d_loss = d_real_loss + d_fake_loss
                
                # Backpropagate and optimize
                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Generate fake images
                fake_image = self.generator(condition)

                # Train the generator
                g_output = self.discriminator({"image": fake_image, "condition":condition})
                g_loss = self.g_loss(g_output, real_label)
                
                # Backpropagate and optimize
                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                if i % 100 == 0:
                    print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, num_epochs, i, len(self.data_loader), d_loss.item(), g_loss.item()))

                if (epoch % 20 == 0) or (epoch == num_epochs - 1):
                    torch.save(self.generator.state_dict(), './checkpoints/generator_%d.pth' % epoch)
                
                fixed_condition = torch.zeros(10,6)
                fixed_condition[:, 2] = 1 # set gender to male
                fixed_condition[:, 4] = 1 # set skin type to dark
                fixed_condition = fixed_condition.to(self.device)
                with torch.no_grad():
                        self.generator.eval()
                        generated_images = self.generator(fixed_condition) 
                        self.generator.train()
                        torch.save_image(generated_images, "./samples/image_epoch_%d.png" % epoch, nrow=5)

                    




def main():
    root_dir = '/home/bmw/Desktop/Projects/gans/GANs/dataset/Hands/Hands'
    csv_file = '/home/bmw/Desktop/Projects/gans/GANs/dataset/HandInfo.csv'

    cgan = CGAN(root_dir, csv_file, batch_size=16, g_learning_rate=0.001, d_learning_rate=0.001)
    cgan.train(1)


if __name__ == '__main__':
    main()
