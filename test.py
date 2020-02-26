import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os
import scipy
from lfw_utils import loaddata, dists_embeddings, img_pairs2embeddings, lfwROC, compute_distances
from models.model import load_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet50 = load_model('resnet50', pretrained=True)
mobileface = load_model('mobilefacenet', pretrained=True)
facenet = load_model('facenet', pretrained=True)


def evallfw(net, device):
    dataname = './data/{}_lfw.mat'.format(net.__class__.__name__)
    if os.path.exists(dataname):
        result = scipy.io.loadmat(dataname)
    else:
        lfwdataset, lfwdataloader = loaddata('E:\data\lfw\pairs.txt',
                                             "./data/lfw_align_112/")
        embL, embR, matchs = img_pairs2embeddings(lfwdataset, lfwdataloader, net, device)
        result = {'fl': embL, 'fr': embR, 'flag': matchs}
        scipy.io.savemat(dataname, result)
    dists = dists_embeddings(result['fl'], result['fr'])
    auc, th, acc = lfwROC(result['flag'], dists)
    print(auc, th, acc)


def lfw_cwc(net, topk=20):
    dataname = './data/{}_lfw_train_test.mat'.format(net.__class__.__name__)
    if not os.path.exists(dataname):
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = datasets.ImageFolder('./data/lfw_align_112', transform=trans)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        train = []
        test = []
        v = np.zeros((6000))
        y = np.zeros(len(dataset), dtype=int)
        for i in tqdm(range(13233)):
            a = dataset[i][1]
            y[i] = a
            if v[a] == 0:
                train.append(i)
                v[a] = 1
            else:
                test.append(i)

        embs = None
        for img, label in tqdm(dataloader):
            img = img.to(device)
            emb = net(img).data.cpu().numpy()
            if embs is None:
                embs = emb
            else:
                embs = np.vstack((embs, emb))
        train_emb = embs[train]
        test_emb = embs[test]
        y_train = y[train]
        y_test = y[test]
        result = {'x_train': train_emb, 'x_test': test_emb, 'y_train': y_train, 'y_test': y_test}
        scipy.io.savemat(dataname, result)
    else:
        result = scipy.io.loadmat(dataname)
        train_emb = result['x_train']
        test_emb = result['x_test']
        y_test = np.squeeze(result['y_test'])

    dists = compute_distances(test_emb, train_emb)
    index_mat = np.argsort(dists, axis=1)

    n = len(y_test)
    acc = []
    remain_idx = np.array(index_mat)
    remain_y = np.array(y_test)
    for i in tqdm(range(topk)):
        index = np.where(remain_idx[:, i] != remain_y)[0]
        acc.append(1 - len(index) / n)
        remain_idx = remain_idx[index]
        remain_y = remain_y[index]
    return acc


print(lfw_cwc(mobileface))

evallfw(resnet50, device)
evallfw(mobileface, device)
evallfw(facenet, device)
