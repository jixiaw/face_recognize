import torch
from lfw_utils import loaddata, loadmodel, align_crop
from src.mtcnn import MTCNN
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
arcface = loadmodel('arcface', 'E:\download\BaiduNetdiskDownload\model_ir_se50.pth')
mtcnn = MTCNN(device=device)

imgdataset = ImageFolder('./data/facebank')
imgs = []
transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
labels = []
for img, label in imgdataset:
    boxes, landmarks = mtcnn.detect_faces(img, return_landmarks=True)
    align = align_crop(img, landmarks, imgsize=112)
    align = transf(align)
    imgs.append(align)
    labels.append(label)
imgs = torch.stack(imgs)

embs = arcface(imgs.to(device)).data.cpu().numpy()

idx_to_class = {}
for a in imgdataset.class_to_idx:
    idx_to_class[imgdataset.class_to_idx[a]] = a
name_labels = [idx_to_class[i] for i in labels]

people = pd.DataFrame(embs)
people['id'] = name_labels
people.to_csv('./data/people.csv', index=False)