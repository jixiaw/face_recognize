from mtcnn_pytorch.mtcnn import MTCNN
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from lfw_utils import align_crop
from torchvision import transforms
from models.model import load_model


def load_vec(df):
    vec = np.array(df.iloc[:, :-1])
    name = df.id.tolist()
    return vec, name


def cmp_vec(vec, embeddings, threshold=1.22):
    dist = (np.sum((embeddings - vec) ** 2, axis=1)) ** 0.5
    index = np.argmin(dist)
    if dist[index] < threshold:
        return index
    else:
        return -1


class FaceRecognizePipeline(object):
    def __init__(self, model='resnet50', device=None, facevec='people.csv'):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.mtcnn = MTCNN(device=self.device)
        self.net = load_model(model, device=self.device, pretrained=True)
        self.vec, self.name = load_vec(pd.read_csv(facevec))

    def forward(self, img_cv2, detect_only=False):
        image = Image.fromarray(img_cv2[..., ::-1])
        if detect_only:
            boxes, landmarks = self.mtcnn.detect_faces(image,  min_face_size=50.0)
            if len(boxes) > 0:
                boxes = np.array(boxes, dtype=np.int)
                for box in boxes:
                    # 画出人脸框
                    img_cv2 = cv2.rectangle(img_cv2, (box[0], box[1], box[2] - box[0], box[3] - box[1]), color=(0, 0, 255))
        else:
            faces, boxes = self.mtcnn.get_align_faces(image, return_boxes=True, return_tensor=True, min_face_size=50.0)
            if boxes is not None and boxes != []:
                boxes = np.array(boxes, dtype=np.int)
                embeddings = self.net(faces.to(self.device)).data.cpu().numpy()
                n = embeddings.shape[0]
                for i in range(n):
                    box = boxes[i]
                    # 画出人脸框
                    img_cv2 = cv2.rectangle(img_cv2, (box[0], box[1], box[2] - box[0], box[3] - box[1]), color=(0, 0, 255))
                    # 人脸识别比较
                    index = cmp_vec(embeddings[i], self.vec)
                    # 画出识别结果
                    if index == -1:
                        img_cv2 = cv2.putText(img_cv2, 'unknown', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 255), 2, cv2.LINE_AA)
                    else:
                        img_cv2 = cv2.putText(img_cv2, self.name[index], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 255), 2, cv2.LINE_AA)
        return img_cv2