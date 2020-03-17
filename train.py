import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from models.insightface import Backbone, MobileFaceNet, Arcface
from lfw_utils import LFWdataset, cal_10fold_acc, dists_embeddings, img_pairs2embeddings
import numpy as np
import os
from tqdm import tqdm


class learner(object):
    def __init__(self, arg):
        self.device = torch.device(arg.device)
        self.model_name = arg.backbone
        if arg.backbone == 'mobilefacenet':
            self.backbone = MobileFaceNet(embedding_size=512).to(self.device)
        else:
            self.backbone = Backbone(50, 0.6, 'ir_se').to(self.device)
        self.step = arg.step
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
        print("loading train data...")
        self.traindataset = datasets.ImageFolder(arg.train_img_path, transform=self.transf)
        self.traindataloader = DataLoader(self.traindataset, batch_size=64, shuffle=True)
        print("loading test data...")
        self.testdataset = LFWdataset(arg.test_pair_path, arg.test_img_path, transform=self.transf)
        self.testdataloader = DataLoader(self.testdataset, batch_size=64, shuffle=False)
        self.load_state()

    def train(self, steps=20):
        max_acc = 0
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
            if (step + 1) % 10 == 0:
                acc = np.mean(self.eval())
                print("step: {}, acc: {}".format(self.step, acc))
            self.step += 1
            # if acc > max_acc:
            #     max_acc = acc
            #     torch.save(self.backbone.state_dict(), 'model_step{}'.format(step))
        self.save_state()

    def eval(self):
        print('Evaluating...')
        embs1, embs2, y_true = img_pairs2embeddings(self.testdataset, self.testdataloader, self.backbone, self.device)
        dists = dists_embeddings(embs1, embs2)
        acc, threshold = cal_10fold_acc(dists, y_true)
        return acc

    def save_state(self, save_path="./weights", model_only=False):
        torch.save(self.backbone.state_dict(), save_path + "/{}_{}".format(self.model_name, self.step))
        if not model_only:
            torch.save(self.margin.state_dict(), save_path + "./margin_{}".format(self.step))
            torch.save(self.optimizer.state_dict(), save_path + "optimizer_{}".format(self.step))

    def load_state(self, save_path="./weights", model_only=False):
        print("loading state...")
        model_path = save_path + "/{}_{}".format(self.model_name, self.step)
        if os.path.exists(model_path):
            self.backbone.load_state_dict(torch.load(model_path))
        if not model_only:
            margin_path = save_path + "./margin_{}".format(self.step)
            optimizer_path = save_path + "optimizer_{}".format(self.step)
            if os.path.exists(margin_path):
                self.margin.load_state_dict(torch.load(margin_path))
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path))


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='train')
    parse.add_argument('--device', type=str, default='cuda:0')
    parse.add_argument('--embedding_size', type=int, default=512)
    parse.add_argument('--class_num', type=int, default=10575)  # 5749
    parse.add_argument('--backbone', type=str, default='mobilefacenet')
    parse.add_argument('--train_img_path', type=str, default='E:\download\BaiduNetdiskDownload\webface_align_112')
    parse.add_argument('--test_pair_path', type=str, default='E:\data\lfw\pairs.txt')
    parse.add_argument('--test_img_path', type=str, default='D:/program/jupyter/computervision/final/data/LFW/lfw_align_112/')
    parse.add_argument('--step', type=int, default=0)
    learn = learner(parse.parse_args())
    print(learn.eval())
    learn.train()
