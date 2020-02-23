import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from tqdm import tqdm
from models.insightface import Backbone, MobileFaceNet
import os
from models.inception_resnet_v1 import InceptionResnetV1
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from skimage import transform as transf


class LFWdataset(Dataset):
    def __init__(self, path, data_dir, transform, img_size=112):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.match = []
        self.img1 = []
        self.img2 = []
        with open(path, 'r') as f:
            file = f.readlines()
            pairs = file[1:]
            print(len(file))
        for pair in pairs:
            pair = pair.split('\n')[0].split('\t')
            if len(pair) == 3:
                name1 = self.data_dir + pair[0] + '/' + pair[0] + '_' + pair[1].zfill(4) + '.jpg'
                name2 = self.data_dir + pair[0] + '/' + pair[0] + '_' + pair[2].zfill(4) + '.jpg'
                match = 1
            else:
                name1 = self.data_dir + pair[0] + '/' + pair[0] + '_' + pair[1].zfill(4) + '.jpg'
                name2 = self.data_dir + pair[2] + '/' + pair[2] + '_' + pair[3].zfill(4) + '.jpg'
                match = 0
            self.match.append(match)
            self.img1.append(name1)
            self.img2.append(name2)

    def __len__(self):
        return len(self.img1)

    def __getitem__(self, index):
        name1 = self.img1[index]
        name2 = self.img2[index]
        img1 = Image.open(name1)
        img2 = Image.open(name2)
        if self.img_size != 112:
            img1 = img1.resize((self.img_size, self.img_size))
            img2 = img2.resize((self.img_size, self.img_size))
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, self.match[index]


def loadmodel(model, model_path, pairs_path=None, data_dir=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_size = 112
    if model == 'facenet':
        net = InceptionResnetV1(pretrained='vggface2', classify=False, device=device).eval()
        img_size = 160
    elif model == 'arcface':
        net = Backbone(50, 0.6, 'ir_se').to(device).eval()
        net.load_state_dict(torch.load(model_path))
    elif model == 'mobile':
        net = MobileFaceNet(embedding_size=512).to(device).eval()
        net.load_state_dict(torch.load(model_path))
    else:
        print(model, 'is not available')
    if pairs_path is None:
        return net
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        lfw = LFWdataset(pairs_path, data_dir, transform, img_size)
        dataloader = DataLoader(lfw, batch_size=32, shuffle=False)
        return net, lfw, dataloader


def loaddata(pairs_path=None, data_dir=None, img_size=112, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    lfw = LFWdataset(pairs_path, data_dir, transform, img_size)
    dataloader = DataLoader(lfw, batch_size=batch_size, shuffle=False)
    return lfw, dataloader


def img_pairs2embeddings(dataset, dataloader, net, device):
    embs1 = None
    embs2 = None
    # net = net.to(device)
    with torch.no_grad():
        for img1, img2, _ in tqdm(dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            emb1 = net(img1).data.cpu().numpy()
            emb2 = net(img2).data.cpu().numpy()
            if embs1 is None:
                embs1 = np.array(emb1)
            else:
                embs1 = np.vstack((embs1, emb1))
            if embs2 is None:
                embs2 = np.array(emb2)
            else:
                embs2 = np.vstack((embs2, emb2))
    return embs1, embs2, np.array(dataset.match)


def dists_embeddings(embL, embR, method=None):
    if len(embL.shape) == 1:
        embL = embL.reshape((1, -1))
    if len(embR.shape) == 1:
        embR = embR.reshape((1, -1))
    assert embL.shape[1] == embR.shape[1]
    if method == 'cos':
        dists = np.sum(embL * embR, axis=1) / (np.linalg.norm(embL, axis=1) * np.linalg.norm(embR, axis=1))
    else:
        dists = (np.sum((embL - embR) ** 2, axis=1)) ** 0.5
    return dists


def getAccuracy(y_true, scores, threshold):
    assert len(y_true) == len(scores)
    y_pred = scores < threshold
    correct = np.sum(y_pred == y_true)
    # p = np.sum(scores[flags == 0] > threshold)
    # n = np.sum(scores[flags == 1] < threshold)
    return correct / len(y_true)


def ROC(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    tpr, fpr, threshold = roc_curve(y_true, y_pred)
    return tpr, fpr, threshold


def lfwROC(y_true, y_pred):
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    tpr, fpr, threshold = ROC(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    maxacc = 0.0
    for t in threshold:
        now = getAccuracy(y_true, y_pred, t)
        if now > maxacc:
            best_threshold = t
            maxacc = now
    return auc_score, best_threshold, maxacc


def cal_10fold_acc(dists, label, k_fold=10):
    dists = np.squeeze(dists)
    label = np.squeeze(label)
    assert len(dists) == len(label)
    acc, threshold = [], []
    nums = len(dists)
    ths = np.arange(0, 2, 0.01)
    kfold = KFold(n_splits=k_fold)
    for train, test in kfold.split(np.arange(nums)):
        traindist = dists[train]
        testdist = dists[test]
        trainy = label[train]
        testy = label[test]
        best_acc = 0
        best_threshold = 0
        for th in ths:
            now_acc = getAccuracy(trainy, traindist, th)
            if now_acc > best_acc:
                best_acc = now_acc
                best_threshold = th
        acc.append(getAccuracy(testy, testdist, best_threshold))
        threshold.append(best_threshold)
    return acc, threshold


def compute_distances(A, B):
    # 计算 A 中每个向量与 B 每个向量的距离
    # input: A: m * k, B: n * k
    # output: m * n
    
    m = np.shape(A)[0]
    n = np.shape(B)[0]
    M = np.dot(A, B.T)
    H = np.tile(np.matrix(np.square(A).sum(axis=1)).T,(1,n))
    K = np.tile(np.matrix(np.square(B).sum(axis=1)),(m,1))
    return np.sqrt(-2 * M + H + K)


# 人脸对齐裁剪
def align_crop(img, landmarks, imgsize=None):
    src = np.array([
        [30.29459953,  51.69630051],
        [65.53179932,  51.50139999],
        [48.02519989,  71.73660278],
        [33.54930115,  92.3655014],
        [62.72990036,  92.20410156]], dtype=np.float32)
    crop_size = np.array((96, 112))
    if imgsize == 112:
        size_diff = imgsize - crop_size
        src += size_diff / 2
        crop_size += size_diff
    facial5points = np.array(landmarks, dtype=np.float32)
    dst = np.squeeze(facial5points)
    if len(dst.shape) == 1:
        facial5points = np.array([[dst[j],dst[j+5]] for j in range(5)])
    # else:
    #     facial5points = np.array(landmarks, dtype=np.float32)
    tform = transf.SimilarityTransform()
    tform.estimate(facial5points, src)
    M = tform.params[0:2,:]
    img_cv2 = np.array(img)[..., ::-1]
    warped = cv2.warpAffine(img_cv2, M, (crop_size[0], crop_size[1]), borderValue=0.0)
    warped = Image.fromarray(warped[..., ::-1])
    return warped