import torch
from lfw_utils import loaddata, loadmodel, align_crop
from mtcnn_pytorch.mtcnn import MTCNN
from models.model import load_model
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from detect_pipeline import FaceRecognizePipeline
from pathlib import Path
from PIL import Image
from database import Database
import numpy as np


def save_data_to_database():
    device = torch.device('cuda:0')
    facerecgnize = FaceRecognizePipeline(device=device)
    facebankpath = Path('./data/facebank')
    db = Database()
    names = []
    pic_names = []
    embs = []
    for path in facebankpath.iterdir():
        if path.is_file():
            continue
        else:

            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                    except:
                        continue
                    emb = facerecgnize.img2embedding(img)
                    emb_str = ','.join([str(i) for i in emb[0]])
                    embs.append(emb)
                    names.append(str(path.name))
                    pic_names.append(file.name)
                    db.add_data(name=path.name, pic_name=file.name, embedding=emb_str)
    return names, pic_names, embs





# 将人脸向量存入csv文件, 已废弃
def prepare_data_csv():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arcface = load_model('resnet50', pretrained=True)
    mtcnn = MTCNN(device=device)

    imgdataset = ImageFolder('./data/facebank')
    imgs = []
    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    labels = []
    for img, label in imgdataset:
        boxes, landmarks = mtcnn.detect_faces(img)
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


if __name__ == '__main__':
    prepare_data_database()