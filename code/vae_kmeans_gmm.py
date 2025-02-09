import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

device = "cuda" if torch.cuda.is_available() else "cpu"

class ThermalDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx]['files'])
        image = Image.open(img_name).convert("RGB")
        label = int(self.data.iloc[idx]['Labels'])  # Binary: 0 or 1

        if self.transform:
            image = self.transform(image)

        return image, label


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

csv_path = "/file.csv"
image_dir = "/images"
dataset = ThermalDataset(csv_path, image_dir, transform)


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)


class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=32*64*64, zDim=32):
        super(VAE, self).__init__()
        # Encoder
        self.encConv1 = nn.Conv2d(imgChannels, 16, 5, padding=2) 
        self.encPool1 = nn.MaxPool2d(2, 2)
        self.encConv2 = nn.Conv2d(16, 32, 5, padding=2)  
        self.encPool2 = nn.MaxPool2d(2, 2)
        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Decoder
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 3, stride=2, padding=1)

    def encoder(self, x):
        x = F.relu(self.encConv1(x))
        x1_size = x.size()
        x = self.encPool1(x)
        x = F.relu(self.encConv2(x))
        x2_size = x.size()
        x = self.encPool2(x)
        x = x.view(-1, 32*64*64)
        mu, logVar = self.encFC1(x), self.encFC2(x)
        return mu, logVar, x2_size, x1_size

    def reparameterize(self, mu, logVar):
        std = torch.exp(logVar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoder(self, z, x2_size, x1_size):
        x = F.relu(self.decFC1(z))
        x = x.view(-1, 32, 64, 64)
        x = F.relu(self.decConv1(x, output_size=x2_size))
        x = torch.sigmoid(self.decConv2(x, output_size=x1_size))
        return x

    def forward(self, x):
        mu, logVar, x2_size, x1_size = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z, x2_size, x1_size)
        return out, mu, logVar, z

def train_vae(model, dataloader, num_epochs=50, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            out, mu, logVar, _ = model(imgs)
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model

# Initialize and Train
vae_model = VAE().to(device)
vae_model = train_vae(vae_model, train_dataloader)


def extract_features(model, dataloader):
    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            _, _, _, z = model(imgs)
            features.append(z.cpu().numpy())
            labels.extend(lbls.numpy())
    return np.array(features).squeeze(), np.array(labels)

train_features, train_labels = extract_features(vae_model, train_dataloader)
val_features, val_labels = extract_features(vae_model, val_dataloader)


kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(train_features)
kmeans_preds = kmeans.predict(val_features)
kmeans_acc = accuracy_score(val_labels, kmeans_preds)
print(f"K-Means Accuracy: {kmeans_acc:.4f}")


gmm = GaussianMixture(n_components=2, random_state=0)
gmm.fit(train_features)
gmm_preds = gmm.predict(val_features)
gmm_acc = accuracy_score(val_labels, gmm_preds)
print(f"GMM Accuracy: {gmm_acc:.4f}")
