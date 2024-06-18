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
    parser.add_argument("--video_path","-e",type=str,default="test_video/brocolli.mp4")
    parser.add_argument("--checkpoint_path","-c",type=str,default="trained_models/vegetables/best.pt")
    parser.add_argument("--pretrained_checkpoint_path", "-p", type=str, default=None)
    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("mps")
    categories = ["Bean","Bitter_Gourd","Bottle_Gourd","Brinjal","Broccoli","Cabbage","Capsicum","Carrot","Cauliflower","Cucumber","Papaya","Potato","Pumpkin","Radish","Tomato"]

    model = resnet50(weights=None)
    model.fc = nn.Linear(in_features=2048, out_features=15)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    softmax = nn.Softmax(dim=1)

    cap = cv2.VideoCapture(args.video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("result.mp4",cv2.VideoWriter_fourcc(*'mp4v'),int(cap.get(cv2.CAP_PROP_FPS)),(width,height))
    with torch.no_grad():
        while cap.isOpened():
            flag,frame = cap.read()
            if not flag:
                break

            image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image = cv2.resize(image,(args.image_size,args.image_size))
            image = np.transpose(image,(2,0,1))
            image = image / 255
            image = np.expand_dims(image,axis=0)

            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

            image = (image - mean) / std

            image = torch.from_numpy(image).float()
            image = image.to(device)

            prediction = model(image)
            prob = softmax(prediction)

            max_value,max_index = torch.max(prob,dim=1)

            cv2.putText(frame, "{}({:0.4f})".format(categories[max_index[0]], max_value[0]), (50, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    args = make_args()
    train(args)