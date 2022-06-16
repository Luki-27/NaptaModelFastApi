import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary            
import torchvision  # for getting the summary of our model
import glob
#data_dir = "C:\\Users\\user\\Desktop\\DataSet"
#train_dir = data_dir + "/train"
#valid_dir = data_dir + "/valid"
#diseases = os.listdir(train_dir)

# datasets for validation and training
#train = ImageFolder(train_dir, transform=transforms.ToTensor())
#valid = ImageFolder(valid_dir, transform=transforms.ToTensor())

labels =\
    ['Apple___Apple_scab',
     'Apple___Black_rot',
     'Apple___Cedar_apple_rust',
     'Apple___healthy',
     'Blueberry___healthy',
     'Cherry_(including_sour)___healthy',
     'Cherry_(including_sour)___Powdery_mildew',
     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
     'Corn_(maize)___Common_rust_',
     'Corn_(maize)___healthy',
     'Corn_(maize)___Northern_Leaf_Blight',
     'Grape___Black_rot',
     'Grape___Esca_(Black_Measles)',
     'Grape___healthy',
     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
     'Orange___Haunglongbing_(Citrus_greening)',
     'Peach___Bacterial_spot',
     'Peach___healthy',
     'Pepper,_bell___Bacterial_spot',
     'Pepper,_bell___healthy',
     'Potato___Early_blight',
     'Potato___healthy',
     'Potato___Late_blight',
     'Raspberry___healthy',
     'Soybean___healthy',
     'Squash___Powdery_mildew',
     'Strawberry___healthy',
     'Strawberry___Leaf_scorch',
     'Tomato___Bacterial_spot',
     'Tomato___Early_blight',
     'Tomato___healthy',
     'Tomato___Late_blight',
     'Tomato___Leaf_Mold',
     'Tomato___Septoria_leaf_spot',
     'Tomato___Spider_mites Two-spotted_spider_mite',
     'Tomato___Target_Spot',
     'Tomato___Tomato_mosaic_virus',
     'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
     ]

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available:
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# for moving data to device (CPU or GPU)
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input

class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
# Architecture for training

# convolution block with BatchNormalization
def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# resnet architecture 
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out        

def predict_image(img, model):
    """Converts image to array and return the predicted class
        with highest probability"""
    # Convert to a batch of 1
    device = get_default_device()

    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    print(preds[0].item())
    return labels[preds[0].item()]
def getfileslist():
    base_dir = r"images"
    ls = [file for file in glob.iglob(f"{base_dir}/*" , recursive= True)] 
    return ls 
def run():

    device = get_default_device()
    model = to_device(ResNet9(3,38), device)
    base_dir = r"images"

    model.load_state_dict(torch.load("plant-disease-model.pth"))
    test = ImageFolder(base_dir, transform=transforms.ToTensor())
    img , lbl = test[0]
    test_images = sorted(os.listdir(base_dir + '/img'))
    i = 0 
    result = []
    for example in test:
        img , lbl = example

        result.append(predict_image(img, model))  
        os.remove(f"{base_dir}/img/{test_images[i]}")
        i+=1
    #print(len(train.classes))
    return result 
if __name__ == "__main__":

    device = get_default_device()
    model = to_device(ResNet9(3, 38), device)
    base_dir = r"images"

    model.load_state_dict(torch.load("plant-disease-model.pth"))
    test = ImageFolder(base_dir, transform=transforms.ToTensor())
    img , lbl = test[0]
    test_images = sorted(os.listdir(base_dir + '/img'))
    i = 0 
    for example in test:
        img , lbl = example
        print(predict_image(img, model))
        os.remove(f"{base_dir}/img/{test_images[i]}")
        i+=1

