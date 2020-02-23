import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from models.insightface import Backbone, MobileFaceNet, Arcface
from lfw_utils import LFWdataset, cal_10fold_acc, dists_embeddings, img_pairs2embeddings
import numpy as np
from tqdm import tqdm


class learner(object):
    def __init__(self, arg):
        self.device = torch.device(arg.device)
        if arg.backbone == 'mobilefacenet':
            self.backbone = MobileFaceNet(embedding_size=512).to(self.device)
        else:
            self.backbone = Backbone(50, 0.6, 'ir_se').to(self.device)
        self.margin = Arcface(embedding_size=arg.embedding_size, classnum=arg.class_num).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD([
                {'params': self.backbone.parameters(), 'weight_decay': 5e-4},
                {'params': self.margin.parameters(), 'weight_decay': 5e-4}
            ], lr=0.1, momentum=0.9, nesterov=True)
        self.transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.traindataset = datasets.ImageFolder(arg.train_img_path, transform=self.transf)
        self.traindataloader = DataLoader(self.traindataset, batch_size=32, shuffle=True)
        self.testdataset = LFWdataset(arg.test_pair_path, arg.test_img_path, transform=self.transf)
        self.testdataloader = DataLoader(self.testdataset, batch_size=32, shuffle=False)

    def train(self, steps=20):
        num_batch = len(self.traindataloader)
        for step in range(steps):
            for i, (x, y) in enumerate(self.traindataloader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                emb = self.backbone(x)
                out = self.margin(emb, y)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                if (i+1) % 100 == 0:
                    print('step: {}/{}, batch: {}/{}, loss: {}'.format(step+1, steps, i+1, num_batch, loss.item()))
            acc = np.mean(self.eval())
            print(acc)

    def eval(self):
        print('Evaluating...')
        embs1, embs2, y_true = img_pairs2embeddings(self.testdataset, self.testdataloader, self.backbone, self.device)
        dists = dists_embeddings(embs1, embs2)
        acc, threshold = cal_10fold_acc(dists, y_true)
        return acc





if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='train')
    parse.add_argument('--device', type=str, default='cuda:0')
    parse.add_argument('--embedding_size', type=int, default=512)
    parse.add_argument('--class_num', type=int, default=5749)
    parse.add_argument('--backbone', type=str, default='mobilefacenet')
    parse.add_argument('--train_img_path', type=str, default='./data/lfw_align_112')
    parse.add_argument('--test_pair_path', type=str, default='E:\data\lfw\pairs.txt')
    parse.add_argument('--test_img_path', type=str, default='./data/lfw_align_112/')
    learn = learner(parse.parse_args())
    print(learn.eval())
    learn.train()
