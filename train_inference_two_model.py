# Import necessary libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from torch.utils.data import random_split
from thop import profile
 
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2

import random

# Create a custom dataset class for test_data and test_labels
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[index]
        return img, label

def configuration_algorithm(_T_paras, time, batch_size, property_1):
    Lowest_MACs = 794
    Highest_MACs = 8637
    # Highest_MACs = 1311778816 +  557935104.0*2
    C_t = random.randint(Lowest_MACs*batch_size,Highest_MACs*batch_size )
    # C_t = Highest_MACs*batch_size 
    Retraining_configuration_list = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])
    Inference_configuration_list = np.asarray([20,24,28,32])
    Retraining_configuration_A_list =  np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])
    # The MACs required for a single sample, including MACs of inference on the teacher model and retraining on the student model.
    Retraining_configuration_C_list =  np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])
    for index in range(len(Retraining_configuration_C_list)):
        if Retraining_configuration_C_list[index]<=1:
            Retraining_configuration_C_list[index] = Retraining_configuration_C_list[index]*(Highest_MACs +  Lowest_MACs*2)
        else:    
            Retraining_configuration_C_list[index] = Highest_MACs +  Retraining_configuration_C_list[index]*(Lowest_MACs*2)
    # print(Retraining_configuration_C_list)
    Inference_configuration_A_list = np.asarray([0.4493, 0.5938, 0.7329, 0.7957])
    Inference_configuration_C_list = np.asarray([635, 671, 745, 794])
    M = 6
    N = 4
    if property_1 is False:
        Inference_configuration_list = Inference_configuration_list[:-1]
        Inference_configuration_A_list = Inference_configuration_A_list[:-1]
        Inference_configuration_C_list = Inference_configuration_C_list[:-1]
        N = 3
    f_max = Inference_configuration_A_list[-1]
    Inference_configuration_A_list = Inference_configuration_A_list/f_max
     
    # print(Inference_configuration_A_list)
    # print(Inference_configuration_C_list)
    L = 0.01
    A_I_min = Inference_configuration_A_list[0]
    if time==1:
        W = f_max/L - Retraining_configuration_A_list[-1]
    else:
        W = f_max/L
    V = A_I_min*_T_paras 
    U = C_t/batch_size
    # print(V,W)
    Best_configuration = [-M,-N,0]
    i = 1
    j = N
    while(i<=M and j>=1):
        c = Retraining_configuration_C_list[i-1] + Inference_configuration_C_list[j-1]
        if c<=U:
            # print(c,U)
            a = V*Retraining_configuration_A_list[i-1]+W*Inference_configuration_A_list[j-1]
            if a>Best_configuration[2]:
                Best_configuration = [i,j,a]
                # print(Best_configuration)
            i+=1
        else:
            j-=1
    _T_paras -= 1/time
    if _T_paras<1e-6:
        _T_paras = 0
    # print(_T_paras)
    # print("before return:",Best_configuration)
    return U, _T_paras, Retraining_configuration_list[Best_configuration[0]-1],Inference_configuration_list[Best_configuration[1]-1] 


def main():
    inference_training_accuracy_list = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=1000
    # model
    teacher = resnet50().to(device) 
    student = mobilenet_v2().to(device)    
    Inference_greedy_student = mobilenet_v2().to(device)   
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])  
    CIFAR10C_list = ["original", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]

    test_labels = np.load('CIFAR-10-C/labels.npy')
    avg_rate=0
    for CIFAR10C in CIFAR10C_list:
        if CIFAR10C == "original": 
            test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=transform, download=False)
            total_testdata_counts = test_labels.size/5
        else: 
            test_data = np.load('CIFAR-10-C/'+CIFAR10C+'.npy')
            test_dataset = CustomDataset(test_data, test_labels, transform=transform)
            total_testdata_counts = test_labels.size
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        if (CIFAR10C == "gaussian_noise" or CIFAR10C == "impulse_noise" or CIFAR10C == "shot_noise" or CIFAR10C == "speckle_noise" ):
            property_1 = False
        else:
            property_1 = True

        # Load the model from the models folder
        teacher.load_state_dict(torch.load("models/res50_model.pth", map_location=torch.device('cuda')))
        student.load_state_dict(torch.load("models/mobile_student_model.pth", map_location=torch.device('cpu')))
        Inference_greedy_student.load_state_dict(torch.load("models/mobile_student_model.pth", map_location=torch.device('cpu')))

        # Set the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(student.parameters(), lr=0.03) 

        teacher_correct_counts = 0
        student_correct_counts = 0
        Inference_greedy_student_correct_counts = 0
        teacher_correct_list = []
        student_correct_list = []
        Inference_greedy_student_correct_list = []
        U_counts = 0
        # Training loop
        
        T_paras = 0
        T = len(test_loader)  
        for i in range(1,T):
            T_paras += 1/i 
        for epoch in range(1):  # Train for 1 epoch
            for i, (images, labels) in enumerate(test_loader):
                _U,T_paras, Retraining_configuration, Inference_configuration = configuration_algorithm(T_paras, i+1, batch_size, property_1)
                U_counts += _U
                # print(i+1,Retraining_configuration, Inference_configuration)
                Retrain = 1 # train 1 epoch, small size
                if Retraining_configuration == 0:
                    Retrain = 0 # no train
                elif Retraining_configuration > 1:
                    Retrain = 2 # train many epoch
                else: None

                # ==================Inference choice==================
                resized_images = F.interpolate(images, size=(Inference_configuration, Inference_configuration), mode='bilinear', align_corners=False) 
                resized_images = resized_images.to(device)
                
                # ==================Retraining choice================== 
                if Retrain == 1 :
                    # Create a list of indices and shuffle them
                    indices = list(range(len(images)))
                    np.random.shuffle(indices)
                    
                    # Calculate the number of samples for the smaller dataset
                    num_samples = int(len(images) * Retraining_configuration)
                    
                    # Get the indices for the smaller dataset and the remaining dataset
                    smaller_indices = indices[:num_samples] 
                    
                    # Use the indices to get the smaller images and labels tensors
                    smaller_images = images[smaller_indices] 
                    smaller_images = smaller_images.to(device)  
                
                # ==================Before================== 
                images = images.to(device)
                labels = labels.to(device)

                student.eval() 
                outputs = student(resized_images)   
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item() 
                student_correct_counts+=correct
                total = labels.size(0)
                print('run [{}], Before training Accuracy: {:.2f} %'.format(i + 1, 100 * correct / total))
                
                student_correct_list.append(round(100 * correct / total,2))


                teacher.eval() 
                with torch.no_grad():
                    outputs = teacher(images)   
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item() 
                teacher_correct_counts+=correct

                teacher_correct_list.append(round(100 * correct / total,2))


                Inference_greedy_student.eval() 
                with torch.no_grad():
                    outputs = Inference_greedy_student(images)   
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item() 
                Inference_greedy_student_correct_counts+=correct

                
                Inference_greedy_student_correct_list.append(round(100 * correct / total,2))
                # ==================Workflow==================
                if Retrain ==1:
                    teacher.eval()
                    with torch.no_grad():
                        predicted_labels = teacher(smaller_images).argmax(dim=1)

                    student.train()
                    optimizer.zero_grad()
                    outputs = student(smaller_images)
                    loss = criterion(outputs, predicted_labels)
                    loss.backward()
                    optimizer.step()
                elif Retrain == 2:
                    teacher.eval()
                    with torch.no_grad():
                        predicted_labels = teacher(images).argmax(dim=1)
                    for epoch_train in range(int(Retraining_configuration)):
                        print("epoch", epoch_train)
                        student.train()
                        optimizer.zero_grad()
                        outputs = student(images)
                        loss = criterion(outputs, predicted_labels)
                        loss.backward()
                        optimizer.step()
                else: None
                    
                # ==================After==================
                student.eval()
                # Forward pass
                outputs = student(resized_images)  
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                # print('run [{}], After Training Accuracy: {:.2f} %'.format(i + 1, 100 * correct / total))
        
        print(CIFAR10C)
        print(f'student_correct_rate: {100 * student_correct_counts/total_testdata_counts:.2f}%')
        print(f'teacher_correct_rate: {100 * teacher_correct_counts/total_testdata_counts:.2f}%')
        print(f'Inference_greedy_student_correct_rate: {100 * Inference_greedy_student_correct_counts/total_testdata_counts:.2f}%')
        print("student_average_MACs:", U_counts/T)         
        print("teacher_greedy_MACs:", 8637)         
        print("student_greedy_MACs:", 794)      
        print("teacher_correct_list:", teacher_correct_list)
        print("student_correct_list:", student_correct_list)
        print( "Inference_greedy_student_correct_list：", Inference_greedy_student_correct_list)
        inference_training_accuracy_list.append(round(100* student_correct_counts/total_testdata_counts,2))
    print(inference_training_accuracy_list)

if __name__ == '__main__': 
    main()
    
"""  
output: 
inference_training_accuracy_list: [79.24, 79.06, 52.19, 72.08, 72.35, 67.2, 70.96, 67.51, 68.44, 64.9, 58.99, 75.7, 64.51, 77.23, 73.15, 69.01, 70.46, 71.69, 69.46, 69.69]
"""
