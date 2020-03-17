from mtcnn_pytorch.mtcnn import MTCNN
import torch
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image
from lfw_utils import align_crop
from torchvision import transforms
from models.model import load_model
from detect_pipeline import load_data_from_database
from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("running on: {}".format(device))

mtcnn = MTCNN(device=device)
arcface = load_model('resnet50', device=device, pretrained=True)


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


def take_photo(name):
    cap = cv2.VideoCapture(0)
    windowName = "take_photo"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    path = Path('./data/facebank') / name
    if os.path.exists(path):
        n = len(os.listdir(path))
    else:
        os.mkdir(path)
        n = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = cv2.putText(frame, "Press space key to take photo.", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                                cv2.LINE_AA)

            cv2.imshow(windowName, frame)
            key = cv2.waitKey(1) & 0xff
            if key == ord(' '):
                image = Image.fromarray(frame[..., ::-1])
                faces = mtcnn.get_align_faces(image, return_largest=True)
                if faces is []:
                    print('No face')
                else:
                    n += 1
                    face = np.array(faces[0])[..., ::-1]
                    cv2.imwrite(str(path / (name + '_' + str(n) + '.jpg')), face)
            if key == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


def capture(net):
    # 初始化摄像头
    keep_processing = True
    cap = cv2.VideoCapture(0)
    windowName = "CV_final"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    print("按键Q-结束视频录制")
    flag = True
    # people_vec = pd.read_csv('./people.csv')
    # vec, name = load_vec(people_vec)
    vec, name = load_data_from_database()
    while (cap.isOpened()):
        if keep_processing:
            ret, frame = cap.read()
            if flag:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # 检测人脸
                faces, boxs = mtcnn.get_align_faces(img, return_boxes=True, return_tensor=True)
                if boxs is not None and boxs != []:
                    boxs = np.array(boxs, dtype=np.int)
                    # 将人脸转为特征向量
                    embeddings = net(faces.to(device)).detach().cpu().numpy()
                    n = embeddings.shape[0]
                    for i in range(n):
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
                frame = cv2.putText(frame, "Recognizing", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                                    cv2.LINE_AA)
            else:
                frame = cv2.putText(frame, "Suspend", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                                    cv2.LINE_AA)

            cv2.imshow(windowName, frame)
            # 接收键盘指令
            key = cv2.waitKey(1) & 0xFF
            # Esc 退出
            if key == 27:
                print("Quit Process ")
                keep_processing = False
            # 空格开始、暂停
            elif key == ord(' '):
                print("stop Process ")
                flag = flag ^ 1
        elif keep_processing is False:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    take_photo('jxw')
    capture(arcface)
