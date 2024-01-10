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
# from thop import profile
 
from torchvision.models import resnet50
from torchvision.models import mobilenet_v2

# Create a custom dataset class for test_data and test_labels
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
    
    Inference_configuration_list = np.asarray([20,24,28,32]) 
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    # model = resnet50().to(device) 
    # model.load_state_dict(torch.load("models/res50_model.pth")) 
    model = mobilenet_v2().to(device) 
    model.load_state_dict(torch.load("models/mobile_student_model.pth"))  
 
    CIFAR10C_list = ["original", "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise","zoom_blur"]
    p_20=""
    p_24=""
    p_28=""
    p_32=""
    m_20=0
    m_28=0
    m_28=0
    m_32=0
    for CIFAR10C in CIFAR10C_list:  
        for Inference_configuration in Inference_configuration_list: 
            # Define the transform for test_data
            test_transform = transforms.Compose([ 
                transforms.Resize((Inference_configuration,Inference_configuration)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) 
            if CIFAR10C == "original": 
                test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=test_transform, download=False)
            else:  
                test_data = np.load('CIFAR-10-C/'+CIFAR10C+'.npy') 
                test_labels = np.load('CIFAR-10-C/labels.npy') 
                test_dataset = CustomCIFAR10CDataset(test_data, test_labels, transform=test_transform)
 
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500, shuffle=True)
            # Testing process
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
                c = round((100 * correct / total),2)
            print("Inference_configuration on",CIFAR10C,":",Inference_configuration,"*",Inference_configuration,":" ,c)
            if Inference_configuration==20:
                p_20+="& "+str(c)
                # print(p_8)
            elif Inference_configuration==24:
                p_24+="& "+str(c)
            elif Inference_configuration==28:
                p_28+="& "+str(c)
            elif Inference_configuration==32:
                p_32+="& "+str(c)
            else: None
            # print(f'Test Accuracy: {100 * correct / total:.2f}%')
            if CIFAR10C == "original": 
                None
            else:
                if Inference_configuration==20: 
                    m_20+=c
                elif Inference_configuration==24:
                    m_28+=c
                elif Inference_configuration==28:
                    m_28+=c
                elif Inference_configuration==32:
                    m_32+=c
                else: None
    print(p_20)
    print(p_24)
    print(p_28)
    print(p_32)
    print(round(m_20/19,2))
    print(round(m_28/19,2))
    print(round(m_28/19,2))
    print(round(m_32/19,2)) 

"""
teacher:
image_size, macs, params_list:
[(8, 82180096.0, 23705252.0), (16, 328099840.0, 23705252.0), (24, 737966080.0, 23705252.0), (32, 1311778816.0, 23705252.0)]

student:
image_size, macs, params_list:
[(8, 34919424.0, 11220132.0), (16, 139522560.0, 11220132.0), (24, 313861120.0, 11220132.0), (32, 557935104.0, 11220132.0)]

"""


"""
mobilenetv2:

Inference_configuration on original : 20 * 20 : 44.98
Inference_configuration on original : 24 * 24 : 59.43
Inference_configuration on original : 28 * 28 : 73.29
Inference_configuration on original : 32 * 32 : 79.57
Inference_configuration on brightness : 20 * 20 : 42.6
Inference_configuration on brightness : 24 * 24 : 54.41
Inference_configuration on brightness : 28 * 28 : 67.95
Inference_configuration on brightness : 32 * 32 : 76.0
Inference_configuration on contrast : 20 * 20 : 23.29
Inference_configuration on contrast : 24 * 24 : 28.1
Inference_configuration on contrast : 28 * 28 : 38.34
Inference_configuration on contrast : 32 * 32 : 47.53
Inference_configuration on defocus_blur : 20 * 20 : 40.46
Inference_configuration on defocus_blur : 24 * 24 : 51.27
Inference_configuration on defocus_blur : 28 * 28 : 63.23
Inference_configuration on defocus_blur : 32 * 32 : 71.11
Inference_configuration on elastic_transform : 20 * 20 : 39.26
Inference_configuration on elastic_transform : 24 * 24 : 49.95
Inference_configuration on elastic_transform : 28 * 28 : 62.48
Inference_configuration on elastic_transform : 32 * 32 : 71.9
Inference_configuration on fog : 20 * 20 : 27.65
Inference_configuration on fog : 24 * 24 : 37.43
Inference_configuration on fog : 28 * 28 : 49.69
Inference_configuration on fog : 32 * 32 : 62.76
Inference_configuration on frost : 20 * 20 : 39.46
Inference_configuration on frost : 24 * 24 : 48.5
Inference_configuration on frost : 28 * 28 : 59.18
Inference_configuration on frost : 32 * 32 : 62.7
Inference_configuration on gaussian_blur : 20 * 20 : 38.42
Inference_configuration on gaussian_blur : 24 * 24 : 48.1
Inference_configuration on gaussian_blur : 28 * 28 : 59.23
Inference_configuration on gaussian_blur : 32 * 32 : 67.03
Inference_configuration on gaussian_noise : 20 * 20 : 42.96
Inference_configuration on gaussian_noise : 24 * 24 : 55.72
Inference_configuration on gaussian_noise : 28 * 28 : 64.53
Inference_configuration on gaussian_noise : 32 * 32 : 56.29
Inference_configuration on glass_blur : 20 * 20 : 40.33
Inference_configuration on glass_blur : 24 * 24 : 50.19
Inference_configuration on glass_blur : 28 * 28 : 62.2
Inference_configuration on glass_blur : 32 * 32 : 62.89
Inference_configuration on impulse_noise : 20 * 20 : 41.36
Inference_configuration on impulse_noise : 24 * 24 : 53.03
Inference_configuration on impulse_noise : 28 * 28 : 60.4
Inference_configuration on impulse_noise : 32 * 32 : 57.39
Inference_configuration on jpeg_compression : 20 * 20 : 42.97
Inference_configuration on jpeg_compression : 24 * 24 : 57.11
Inference_configuration on jpeg_compression : 28 * 28 : 69.32
Inference_configuration on jpeg_compression : 32 * 32 : 74.72
Inference_configuration on motion_blur : 20 * 20 : 35.84
Inference_configuration on motion_blur : 24 * 24 : 44.08
Inference_configuration on motion_blur : 28 * 28 : 53.93
Inference_configuration on motion_blur : 32 * 32 : 62.44
Inference_configuration on pixelate : 20 * 20 : 43.03
Inference_configuration on pixelate : 24 * 24 : 55.91
Inference_configuration on pixelate : 28 * 28 : 69.68
Inference_configuration on pixelate : 32 * 32 : 76.98
Inference_configuration on saturate : 20 * 20 : 38.15
Inference_configuration on saturate : 24 * 24 : 50.59
Inference_configuration on saturate : 28 * 28 : 63.69
Inference_configuration on saturate : 32 * 32 : 71.63
Inference_configuration on shot_noise : 20 * 20 : 43.3
Inference_configuration on shot_noise : 24 * 24 : 56.78
Inference_configuration on shot_noise : 28 * 28 : 66.57
Inference_configuration on shot_noise : 32 * 32 : 61.98
Inference_configuration on snow : 20 * 20 : 39.3
Inference_configuration on snow : 24 * 24 : 50.43
Inference_configuration on snow : 28 * 28 : 61.69
Inference_configuration on snow : 32 * 32 : 65.82
Inference_configuration on spatter : 20 * 20 : 41.38
Inference_configuration on spatter : 24 * 24 : 55.42
Inference_configuration on spatter : 28 * 28 : 67.33
Inference_configuration on spatter : 32 * 32 : 71.91
Inference_configuration on speckle_noise : 20 * 20 : 43.1
Inference_configuration on speckle_noise : 24 * 24 : 56.78
Inference_configuration on speckle_noise : 28 * 28 : 66.68
Inference_configuration on speckle_noise : 32 * 32 : 62.86
Inference_configuration on zoom_blur : 20 * 20 : 41.7
Inference_configuration on zoom_blur : 24 * 24 : 49.96
Inference_configuration on zoom_blur : 28 * 28 : 61.23
Inference_configuration on zoom_blur : 32 * 32 : 67.78
& 44.98& 42.6& 23.29& 40.46& 39.26& 27.65& 39.46& 38.42& 42.96& 40.33& 41.36& 42.97& 35.84& 43.03& 38.15& 43.3& 39.3& 41.38& 43.1& 41.7
& 59.43& 54.41& 28.1& 51.27& 49.95& 37.43& 48.5& 48.1& 55.72& 50.19& 53.03& 57.11& 44.08& 55.91& 50.59& 56.78& 50.43& 55.42& 56.78& 49.96
& 73.29& 67.95& 38.34& 63.23& 62.48& 49.69& 59.18& 59.23& 64.53& 62.2& 60.4& 69.32& 53.93& 69.68& 63.69& 66.57& 61.69& 67.33& 66.68& 61.23
& 79.57& 76.0& 47.53& 71.11& 71.9& 62.76& 62.7& 67.03& 56.29& 62.89& 57.39& 74.72& 62.44& 76.98& 71.63& 61.98& 65.82& 71.91& 62.86& 67.78
39.19
111.64
111.64
65.88

"""
        
"""
resnet50:

Inference_configuration on original : 20 * 20 : 54.52
Inference_configuration on original : 24 * 24 : 71.96
Inference_configuration on original : 28 * 28 : 79.03
Inference_configuration on original : 32 * 32 : 86.14
Inference_configuration on brightness : 20 * 20 : 49.2
Inference_configuration on brightness : 24 * 24 : 66.25
Inference_configuration on brightness : 28 * 28 : 74.19
Inference_configuration on brightness : 32 * 32 : 83.21
Inference_configuration on contrast : 20 * 20 : 32.25
Inference_configuration on contrast : 24 * 24 : 40.69
Inference_configuration on contrast : 28 * 28 : 42.73
Inference_configuration on contrast : 32 * 32 : 55.34
Inference_configuration on defocus_blur : 20 * 20 : 50.71
Inference_configuration on defocus_blur : 24 * 24 : 62.59
Inference_configuration on defocus_blur : 28 * 28 : 66.58
Inference_configuration on defocus_blur : 32 * 32 : 73.97
Inference_configuration on elastic_transform : 20 * 20 : 48.99
Inference_configuration on elastic_transform : 24 * 24 : 61.51
Inference_configuration on elastic_transform : 28 * 28 : 66.78
Inference_configuration on elastic_transform : 32 * 32 : 76.58
Inference_configuration on fog : 20 * 20 : 39.32
Inference_configuration on fog : 24 * 24 : 50.53
Inference_configuration on fog : 28 * 28 : 55.35
Inference_configuration on fog : 32 * 32 : 70.4
Inference_configuration on frost : 20 * 20 : 44.19
Inference_configuration on frost : 24 * 24 : 60.48
Inference_configuration on frost : 28 * 28 : 66.96
Inference_configuration on frost : 32 * 32 : 76.08
Inference_configuration on gaussian_blur : 20 * 20 : 48.99
Inference_configuration on gaussian_blur : 24 * 24 : 58.75
Inference_configuration on gaussian_blur : 28 * 28 : 61.6
Inference_configuration on gaussian_blur : 32 * 32 : 68.41
Inference_configuration on gaussian_noise : 20 * 20 : 52.23
Inference_configuration on gaussian_noise : 24 * 24 : 68.27
Inference_configuration on gaussian_noise : 28 * 28 : 72.88
Inference_configuration on gaussian_noise : 32 * 32 : 72.93
Inference_configuration on glass_blur : 20 * 20 : 49.98
Inference_configuration on glass_blur : 24 * 24 : 62.63
Inference_configuration on glass_blur : 28 * 28 : 68.07
Inference_configuration on glass_blur : 32 * 32 : 70.55
Inference_configuration on impulse_noise : 20 * 20 : 49.99
Inference_configuration on impulse_noise : 24 * 24 : 64.57
Inference_configuration on impulse_noise : 28 * 28 : 66.0
Inference_configuration on impulse_noise : 32 * 32 : 62.43
Inference_configuration on jpeg_compression : 20 * 20 : 53.04
Inference_configuration on jpeg_compression : 24 * 24 : 69.63
Inference_configuration on jpeg_compression : 28 * 28 : 75.72
Inference_configuration on jpeg_compression : 32 * 32 : 82.44
Inference_configuration on motion_blur : 20 * 20 : 45.79
Inference_configuration on motion_blur : 24 * 24 : 54.6
Inference_configuration on motion_blur : 28 * 28 : 56.96
Inference_configuration on motion_blur : 32 * 32 : 66.47
Inference_configuration on pixelate : 20 * 20 : 53.03
Inference_configuration on pixelate : 24 * 24 : 68.5
Inference_configuration on pixelate : 28 * 28 : 75.12
Inference_configuration on pixelate : 32 * 32 : 82.33
Inference_configuration on saturate : 20 * 20 : 46.07
Inference_configuration on saturate : 24 * 24 : 62.08
Inference_configuration on saturate : 28 * 28 : 69.02
Inference_configuration on saturate : 32 * 32 : 78.6
Inference_configuration on shot_noise : 20 * 20 : 52.94
Inference_configuration on shot_noise : 24 * 24 : 69.11
Inference_configuration on shot_noise : 28 * 28 : 74.45
Inference_configuration on shot_noise : 32 * 32 : 76.17
Inference_configuration on snow : 20 * 20 : 45.69
Inference_configuration on snow : 24 * 24 : 61.02
Inference_configuration on snow : 28 * 28 : 69.1
Inference_configuration on snow : 32 * 32 : 76.43
Inference_configuration on spatter : 20 * 20 : 48.92
Inference_configuration on spatter : 24 * 24 : 64.42
Inference_configuration on spatter : 28 * 28 : 70.4
Inference_configuration on spatter : 32 * 32 : 75.45
Inference_configuration on speckle_noise : 20 * 20 : 52.8
Inference_configuration on speckle_noise : 24 * 24 : 68.98
Inference_configuration on speckle_noise : 28 * 28 : 74.13
Inference_configuration on speckle_noise : 32 * 32 : 75.89
Inference_configuration on zoom_blur : 20 * 20 : 52.83
Inference_configuration on zoom_blur : 24 * 24 : 61.98
Inference_configuration on zoom_blur : 28 * 28 : 64.91
Inference_configuration on zoom_blur : 32 * 32 : 70.14
& 54.52& 49.2& 32.25& 50.71& 48.99& 39.32& 44.19& 48.99& 52.23& 49.98& 49.99& 53.04& 45.79& 53.03& 46.07& 52.94& 45.69& 48.92& 52.8& 52.83
& 71.96& 66.25& 40.69& 62.59& 61.51& 50.53& 60.48& 58.75& 68.27& 62.63& 64.57& 69.63& 54.6& 68.5& 62.08& 69.11& 61.02& 64.42& 68.98& 61.98
& 79.03& 74.19& 42.73& 66.58& 66.78& 55.35& 66.96& 61.6& 72.88& 68.07& 66.0& 75.72& 56.96& 75.12& 69.02& 74.45& 69.1& 70.4& 74.13& 64.91
& 86.14& 83.21& 55.34& 73.97& 76.58& 70.4& 76.08& 68.41& 72.93& 70.55& 62.43& 82.44& 66.47& 82.33& 78.6& 76.17& 76.43& 75.45& 75.89& 70.14
48.26
128.82
128.82
73.36
"""
    
            
