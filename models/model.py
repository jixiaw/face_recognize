from .inception_resnet_v1 import InceptionResnetV1
from .insightface import Backbone, MobileFaceNet, Arcface
import torch


def load_model(model_name, device=None, train_mode=False, pretrained=False):
    if device == None:
        device = torch.device('cpu')
    else:
        device = device
    if model_name == 'resnet50':
        net = Backbone(50, 0.6, 'ir_se').to(device)
        model_path = 'E:\download\BaiduNetdiskDownload\model_ir_se50.pth'
    elif model_name == 'mobilefacenet':
        net = MobileFaceNet(embedding_size=512).to(device)
        model_path = 'E:\download\BaiduNetdiskDownload\model_mobilefacenet.pth'
    elif model_name == 'facenet':
        net = InceptionResnetV1(device=device)
        model_path = ''
    else:
        raise Exception('No such model!')
    if train_mode == False:
        net.eval()
    if pretrained:
        net.load_state_dict(torch.load(model_path))
    return net


if __name__ == '__main__':
    net = load_model('mobilefacenet', pretrained=True)
    print(net)