# # Import necessary libraries
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import numpy as np
# from torch.utils.data import Dataset
# from PIL import Image
# from torch.utils.data import ConcatDataset

# import os
# from torchvision.models import resnet50
# from torchvision.models import mobilenet_v2


# class CustomCIFAR10CDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         img, label = self.data[index], self.labels[index]
#         img = Image.fromarray(img)

#         if self.transform:
#             img = self.transform(img)

#         return img, label

# if __name__ == '__main__':

#     # Hyper-parameters
#     num_epochs = 150
#     Inference_configuration_list = np.asarray([20,24,28,32])
#     # CIFAR10C_list = ["original","contrast","defocus_blur","gaussian_noise","jpeg_compression","motion_blur","shot_noise"]
#     CIFAR10C_list = ["original", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]

#     # Define the device, model, loss function, and optimizer
#     device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#     print(device)
#     model = resnet50().to(device)
#     criterion = nn.CrossEntropyLoss()
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
#     optimizer =  torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum = 0.9) 
  
#     training_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4), 
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]) 
     
#     combined_train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, transform=training_transform, download=False)

#     train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=256, shuffle=True, num_workers=2)
    

#     # Training process
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0

#         for i, (images, labels) in enumerate(train_loader):
#             images = images.to(device)
#             labels = labels.to(device)

#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Backward and optimize
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')
#         # Testing process
        
#     for CIFAR10C in CIFAR10C_list:
#         for image_size in Inference_configuration_list:
#             # Load CIFAR-10-C dataset
#             test_transform = transforms.Compose([
#                 transforms.Resize((image_size, image_size)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ]) 

#             if CIFAR10C == "original": 
#                 test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=test_transform, download=False)
#             else:  
#                 test_data = np.load('CIFAR-10-C/'+CIFAR10C+'.npy') 
#                 test_labels = np.load('CIFAR-10-C/labels.npy') 
#                 test_dataset = CustomCIFAR10CDataset(test_data, test_labels, transform=test_transform)
            
#             test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

#             model.eval()
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for images, labels in test_loader:
#                     images = images.to(device)
#                     labels = labels.to(device) 
#                     outputs = model(images)
#                     _, predicted = outputs.max(1)
#                     total += labels.size(0)
#                     correct += predicted.eq(labels).sum().item()
#             print(CIFAR10C)
#             print(f'Test Accuracy: {100 * correct / total:.2f}%')

#     # Check if the models folder exists, if not, create one
#     if not os.path.exists("models"):
#         os.makedirs("models") 
#     # Save the model to the models folder
#     torch.save(model.state_dict(), "models/res50_model"+".pth")  

# """ 
# (chg) [caihuaiguang@earth /home/caihuaiguang/inference_training/EATA/CIFAR]$ python res50_teacher_training.py
# cuda:3
# Epoch [1/30], Loss: 2.0497, Accuracy: 30.92%
# Epoch [2/30], Loss: 1.4095, Accuracy: 48.82%
# Epoch [3/30], Loss: 1.1525, Accuracy: 58.48%
# Epoch [4/30], Loss: 0.9836, Accuracy: 64.94%
# Epoch [5/30], Loss: 0.8569, Accuracy: 69.63%
# Epoch [6/30], Loss: 0.7583, Accuracy: 73.23%
# Epoch [7/30], Loss: 0.6606, Accuracy: 76.82%
# Epoch [8/30], Loss: 0.5917, Accuracy: 79.34%
# Epoch [9/30], Loss: 0.5368, Accuracy: 81.38%
# Epoch [10/30], Loss: 0.4856, Accuracy: 83.21%
# Epoch [11/30], Loss: 0.4456, Accuracy: 84.58%
# Epoch [12/30], Loss: 0.4082, Accuracy: 85.78%
# Epoch [13/30], Loss: 0.3832, Accuracy: 86.70%
# Epoch [14/30], Loss: 0.3517, Accuracy: 87.74%
# Epoch [15/30], Loss: 0.3287, Accuracy: 88.60%
# Epoch [16/30], Loss: 0.3093, Accuracy: 89.23%
# Epoch [17/30], Loss: 0.2903, Accuracy: 89.78%
# Epoch [18/30], Loss: 0.2730, Accuracy: 90.55%
# Epoch [19/30], Loss: 0.2543, Accuracy: 91.15%
# Epoch [20/30], Loss: 0.2401, Accuracy: 91.63%
# Epoch [21/30], Loss: 0.2270, Accuracy: 92.12%
# Epoch [22/30], Loss: 0.2084, Accuracy: 92.65%
# Epoch [23/30], Loss: 0.1993, Accuracy: 93.01%
# Epoch [24/30], Loss: 0.1883, Accuracy: 93.45%
# Epoch [25/30], Loss: 0.1765, Accuracy: 93.89%
# Epoch [26/30], Loss: 0.1638, Accuracy: 94.33%
# Epoch [27/30], Loss: 0.1576, Accuracy: 94.45%
# Epoch [28/30], Loss: 0.1499, Accuracy: 94.76%
# Epoch [29/30], Loss: 0.1399, Accuracy: 95.06%
# Epoch [30/30], Loss: 0.1302, Accuracy: 95.45%
# original
# Test Accuracy: 25.15%
# original
# Test Accuracy: 59.69%
# original
# Test Accuracy: 82.29%
# original
# Test Accuracy: 86.64%
# brightness
# Test Accuracy: 24.02%
# brightness
# Test Accuracy: 56.02%
# brightness
# Test Accuracy: 78.57%
# brightness
# Test Accuracy: 82.94%
# contrast
# Test Accuracy: 15.46%
# contrast
# Test Accuracy: 34.40%
# contrast
# Test Accuracy: 51.87%
# contrast
# Test Accuracy: 62.03%
# defocus_blur
# Test Accuracy: 25.10%
# defocus_blur
# Test Accuracy: 56.90%
# defocus_blur
# Test Accuracy: 73.85%
# defocus_blur
# Test Accuracy: 74.39%
# elastic_transform
# Test Accuracy: 24.69%
# elastic_transform
# Test Accuracy: 54.93%
# elastic_transform
# Test Accuracy: 72.60%
# elastic_transform
# Test Accuracy: 74.31%
# fog
# Test Accuracy: 19.30%
# fog
# Test Accuracy: 41.40%
# fog
# Test Accuracy: 64.09%
# fog
# Test Accuracy: 75.75%
# frost
# Test Accuracy: 19.95%
# frost
# Test Accuracy: 49.22%
# frost
# Test Accuracy: 70.47%
# frost
# Test Accuracy: 69.57%
# gaussian_blur
# Test Accuracy: 24.95%
# gaussian_blur
# Test Accuracy: 55.35%
# gaussian_blur
# Test Accuracy: 69.80%
# gaussian_blur
# Test Accuracy: 67.54%
# gaussian_noise
# Test Accuracy: 25.17%
# gaussian_noise
# Test Accuracy: 56.84%
# gaussian_noise
# Test Accuracy: 74.51%
# gaussian_noise
# Test Accuracy: 50.82%
# glass_blur
# Test Accuracy: 24.84%
# glass_blur
# Test Accuracy: 55.32%
# glass_blur
# Test Accuracy: 69.26%
# glass_blur
# Test Accuracy: 48.16%
# impulse_noise
# Test Accuracy: 25.03%
# impulse_noise
# Test Accuracy: 54.02%
# impulse_noise
# Test Accuracy: 70.01%
# impulse_noise
# Test Accuracy: 57.99%
# jpeg_compression
# Test Accuracy: 25.18%
# jpeg_compression
# Test Accuracy: 58.20%
# jpeg_compression
# Test Accuracy: 78.16%
# jpeg_compression
# Test Accuracy: 75.43%
# motion_blur
# Test Accuracy: 24.63%
# motion_blur
# Test Accuracy: 52.23%
# motion_blur
# Test Accuracy: 64.23%
# motion_blur
# Test Accuracy: 64.71%
# pixelate
# Test Accuracy: 25.15%
# pixelate
# Test Accuracy: 57.85%
# pixelate
# Test Accuracy: 78.10%
# pixelate
# Test Accuracy: 67.97%
# saturate
# Test Accuracy: 23.05%
# saturate
# Test Accuracy: 52.91%
# saturate
# Test Accuracy: 74.92%
# saturate
# Test Accuracy: 79.66%
# shot_noise
# Test Accuracy: 25.24%
# shot_noise
# Test Accuracy: 57.33%
# shot_noise
# Test Accuracy: 76.42%
# shot_noise
# Test Accuracy: 59.45%
# snow
# Test Accuracy: 24.15%
# snow
# Test Accuracy: 51.88%
# snow
# Test Accuracy: 71.14%
# snow
# Test Accuracy: 68.02%
# spatter
# Test Accuracy: 25.09%
# spatter
# Test Accuracy: 55.02%
# spatter
# Test Accuracy: 76.58%
# spatter
# Test Accuracy: 76.28%
# speckle_noise
# Test Accuracy: 25.12%
# speckle_noise
# Test Accuracy: 57.15%
# speckle_noise
# Test Accuracy: 76.30%
# speckle_noise
# Test Accuracy: 61.48%
# zoom_blur
# Test Accuracy: 24.77%
# zoom_blur
# Test Accuracy: 57.81%
# zoom_blur
# Test Accuracy: 72.44%
# zoom_blur
# Test Accuracy: 70.17%
# """


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