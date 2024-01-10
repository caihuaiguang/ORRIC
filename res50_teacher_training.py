


# Import necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import ConcatDataset

import os
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2


class CustomCIFAR10CDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == '__main__':

    # Hyper-parameters
    num_epochs = 100
    Inference_configuration_list = np.asarray([20,24,28,32])
    # CIFAR10C_list = ["original","contrast","defocus_blur","gaussian_noise","jpeg_compression","motion_blur","shot_noise"]
    CIFAR10C_list = ["original", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]

    # Define the device, model, loss function, and optimizer
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = resnet50().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum = 0.9) 
  
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
     
    combined_train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, transform=training_transform, download=False)

    train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=256, shuffle=True, num_workers=2)
    
    # Create validation dataset
    validation_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 

    validation_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=validation_transform, download=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, shuffle=False, num_workers=2)

    # Training process
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

        # Validation process
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device) 
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

        # Update learning rate
        scheduler.step(100 * correct / total)

    # Testing process
    for CIFAR10C in CIFAR10C_list:
        for image_size in Inference_configuration_list:
            # Load CIFAR-10-C dataset
            test_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) 

            if CIFAR10C == "original": 
                test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=test_transform, download=False)
            else:  
                test_data = np.load('CIFAR-10-C/'+CIFAR10C+'.npy') 
                test_labels = np.load('CIFAR-10-C/labels.npy') 
                test_dataset = CustomCIFAR10CDataset(test_data, test_labels, transform=test_transform)
            
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device) 
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            print(CIFAR10C)
            print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Check if the models folder exists, if not, create one
    if not os.path.exists("models"):
        os.makedirs("models") 
    # Save the model to the models folder
    torch.save(model.state_dict(), "models/res50_model"+".pth")
