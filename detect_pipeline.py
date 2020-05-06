from mtcnn_pytorch.mtcnn import MTCNN
import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from lfw_utils import align_crop
from torchvision import transforms
from models.model import load_model
from database import Database


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


def load_data_from_database(return_picnames=False):
    db = Database()
    success, result = db.search_all()
    names = None
    picnames = None
    embs = None
    if success:
        names = [e[1] for e in result]
        if return_picnames:
            picnames = [e[2] for e in result]
        embs_str = [e[3] for e in result]
        embs = np.array([[float(e) for e in emb_str.split(',')] for emb_str in embs_str])
    if return_picnames:
        return embs, names, picnames
    else:
        return embs, names


class FaceRecognizePipeline(object):
    def __init__(self, model='resnet50', device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.mtcnn = MTCNN(device=self.device)
        self.net = load_model(model, device=self.device, pretrained=True)
        # self.vec, self.name = load_vec(pd.read_csv(facevec))
        self.vec, self.name = load_data_from_database()

    def img2embedding(self, image, flip=False):
        '''
        :param image:
        :return: the embedding of biggest face in image
        '''
        face, box = self.mtcnn.get_align_faces(image, return_largest=True, return_boxes=True, return_tensor=True, min_face_size=50.0)
        if box is not None and box != []:
            embedding = self.net(face.to(self.device)).data.cpu().numpy()
        else:
            embedding = []
        return embedding

    def forward(self, image, detect_only=False):
        '''
        :param image: The image of cv2 format or PIL image
        :param detect_only: If true: only detect; else detect and recognize
        :return: The detected image of cv2 format
        '''
        if image is None:
            return None
        if isinstance(image, np.ndarray):
            img_cv2 = np.array(image)
            image = Image.fromarray(img_cv2[..., ::-1])
        else:
            img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # image = Image.fromarray(img_cv2[..., ::-1])
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