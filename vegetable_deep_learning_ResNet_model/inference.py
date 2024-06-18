import torch
import torch.nn as nn
from dataset import VegetableDataset
from model import CNN
from torchvision.transforms import Compose,Resize,ToTensor,RandomAffine,ColorJitter
from torchvision.models import resnet50, ResNet50_Weights
import argparse
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore")

def make_args():
    parser = argparse.ArgumentParser("Test Argument")
    parser.add_argument("--data_path", "-d", type=str, default="Vegetable_Images")
    parser.add_argument("--image_size","-i",type=int,default=224)
    parser.add_argument("--image_path","-e",type=str,default="Vegetable_Images/test/Carrot/1001.jpg")
    parser.add_argument("--checkpoint_path","-c",type=str,default="trained_models/vegetables/best.pt")
    parser.add_argument("--pretrained_checkpoint_path", "-p", type=str, default=None)
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("mps")
    categories = ["Bean","Bitter_Gourd","Bottle_Gourd","Brinjal","Broccoli","Cabbage","Capsicum","Carrot","Cauliflower","Cucumber","Papaya","Potato","Pumpkin","Radish","Tomato"]

    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(args.image_size,args.image_size))
    image = np.transpose(image,(2,0,1))
    image = image / 255
    image = np.expand_dims(image,axis=0)

    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    image = (image - mean) / std

    image = torch.from_numpy(image).float()
    image = image.to(device)

    model = resnet50(weights=None)
    model.fc = nn.Linear(in_features=2048, out_features=15)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        prediction = model(image)
        prob = softmax(prediction)

    max_value,max_index = torch.max(prob,dim=1)
    cv2.imshow("The image is about class: {}.The confident score of: {}".format(categories[max_index[0]],max_value[0]),ori_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    args = make_args()
    train(args)