from torch.utils.data import Dataset,DataLoader
import os
from torchvision.transforms import Compose, Resize, ToTensor
import cv2

class VegetableDataset(Dataset):
    def __init__(self,root,is_train=True,transform=None):
        if is_train:
            data_path = os.path.join(root,"train")
        else:
            data_path = os.path.join(root,"validation")
        self.all_image_paths = []
        self.all_labels = []
        self.categories =["Bean","Bitter_Gourd","Bottle_Gourd","Brinjal","Broccoli","Cabbage","Capsicum","Carrot","Cauliflower","Cucumber","Papaya","Potato","Pumpkin","Radish","Tomato"]
        for index, category in enumerate(self.categories):
            category_path = os.path.join(data_path,category)
            for item in os.listdir(category_path):
                image_path = os.path.join(category_path,item)
                if item.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                    self.all_image_paths.append(image_path)
                    self.all_labels.append(index)
        self.transform = transform

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, item):
        image_path = self.all_image_paths[item]
        image = cv2.imread(image_path)
        if self.transform:
            image = self.transform(image)
        label = self.all_labels[item]

        return image, label

if __name__ == '__main__':
    transform = Compose([
        ToTensor(),
        Resize((224,224))
    ])
    dataset = VegetableDataset(root="Vegetable_Images",is_train=False,transform =transform)
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 16,
        num_workers=8,
        drop_last=True,
        shuffle=True
    )
    for images, labels in dataloader:
        print(images.shape)
        print(labels)