from mtcnn_pytorch.mtcnn import MTCNN
import torch
import numpy as np
import cv2
import sys
import os
import math
import datetime
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from lfw_utils import loadmodel, align_crop
from torchvision import transforms
from models.model import load_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("running on: {}".format(device))

mtcnn = MTCNN(device=device)
# resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
arcface = load_model('resnet50', device=device, pretrained=True)
# mobileface = loadmodel('mobile', )


def img2emb(img, net, img_size=112):
    facesimg = []
    faces = []
    transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    boxes, _, landmarks = mtcnn.detect(img, landmarks=True)
    for landmark in landmarks:
        face = align_crop(img, landmark, 112)
        if img_size != 112:
            face = face.resize((img_size, img_size))
        facesimg.append(face)
        faces.append(transf(face))
    faces = torch.stack(faces)
    embeddings = net(faces.to(device)).detach().cpu().numpy()
    return facesimg, embeddings


def cmp_vec(vec, embeddings, threshold=1.22):
    dist = (np.sum((embeddings - vec) ** 2, axis=1)) ** 0.5
    index = np.argmin(dist)
    if dist[index] < threshold:
        return index
    else:
        return -1


def load_vec(df):
    vec = np.array(df.iloc[:, :-1])
    name = df.id.tolist()
    return vec, name


def process(img, boxes, img_size):
    faces = []
    for box in boxes:
        box = [
            int(max(box[0], 0)),
            int(max(box[1], 0)),
            int(min(box[2], img.size[0])),
            int(min(box[3], img.size[1])),
        ]
        face = img.crop(box).resize((img_size, img_size), 2)
        face = F.to_tensor(np.float32(face))
        face = (face - 127.5) / 128
        faces.append(face)
    faces = torch.stack(faces)
    return faces


# 根据人脸关键点截取人脸, 转为tensor
def process_align(img, landmarks, img_size=112):
    faces = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    for landmark in landmarks:
        align_face = align_crop(img, landmark, imgsize=112)
        if img_size != 112:
            align_face = align_face.resize((img_size, img_size))
        # align_face = transform(align_face)
        faces.append(transform(align_face))
    faces = torch.stack(faces)
    return faces


def addpic(img1, img2, pos):
    rows, cols, channels = img2.shape
    h, w, c = img1.shape
    if pos[0] + rows <= h and pos[1] + cols <= w:
        img1[pos[0]:pos[0]+rows, pos[1]:pos[1]+cols] = img2


def capture(net, show_similar_face=False):
    # 初始化摄像头
    keep_processing = True
    camera_to_use = 0
    cap = cv2.VideoCapture(0)
    windowName = "CV_final"
    frameSize = (640, 480)  # 指定窗口大小
    # 摄像头开启检测
    if not (((len(sys.argv) == 2) and (cap.open(str(sys.argv[1]))))
            or (cap.open(camera_to_use))):
        print("ERROR：No video file specified or camera connected.")
        return -1
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    print("按键Q-结束视频录制")
    flag = True
    people_vec = pd.read_csv('./people.csv')
    vec, name = load_vec(people_vec)
    while (cap.isOpened()):
        if keep_processing:
            ret, frame = cap.read()
            if flag:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 检测人脸
                faces, boxs = mtcnn.get_align_faces(img, return_boxes=True, return_tensor=True)
                # print(boxs, landmarks)
                if boxs is not None and boxs != []:
                    # print(boxs, boxs == [], len(boxs))
                    boxs = np.array(boxs, dtype=np.int)
                    # 将人脸转为特征向量
                    embeddings = net(faces.to(device)).detach().cpu().numpy()
                    n = embeddings.shape[0]
                    for i in range(n):
                        # if probs[i] < 0.9:
                        #     continue
                        # flag = flag ^ 1
                        box = boxs[i]
                        # 画出人脸框
                        frame = cv2.rectangle(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]), color=(0, 0, 255))
                        # 人脸识别比较
                        index = cmp_vec(embeddings[i], vec)
                        # 画出识别结果
                        if index == -1:
                            frame = cv2.putText(frame, 'unknown', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                                (0, 255, 255), 2, cv2.LINE_AA)
                        else:
                            frame = cv2.putText(frame, name[index], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                                (0, 255, 255), 2, cv2.LINE_AA)
                            if show_similar_face:
                                pic_path = './data/facebank/' + name[index]
                                picname = pic_path + '/' + os.listdir(pic_path)[0]
                                # print(picname)
                                similar_pic = cv2.resize(cv2.imread(picname), (60, 60))

                                addpic(frame, similar_pic, (box[1], box[2]))

                frame = cv2.putText(frame, "Recognizing", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                                    cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Suspend", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                                    cv2.LINE_AA)

            cv2.imshow(windowName, frame)
            #             stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # 接收键盘指令
            key = cv2.waitKey(1) & 0xFF
            # Esc 退出
            if (key == 27):
                print("Quit Process ")
                keep_processing = False
            # 空格开始、暂停
            elif key == ord(' '):
                print("stop Process ")
                flag = flag ^ 1
        elif keep_processing == False:
            break
    cap.release()

    cv2.destroyAllWindows()


capture(arcface)
