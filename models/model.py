from models.inception_v1 import InceptionResnetV1
from models.insightface import Backbone, MobileFaceNet, Arcface
import torch
import os


def load_model(model_name, device=None, train_mode=False, pretrained=False):
    work_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(work_dir, 'weight')
    if device is None:
        device = torch.device('cpu')
    else:
        device = device
    if model_name == 'resnet50':
        net = Backbone(50, 0.6, 'ir_se').to(device)
        model_path = os.path.join(model_dir, 'model_ir_se50.pth')
    elif model_name == 'mobilefacenet':
        net = MobileFaceNet(embedding_size=512).to(device)
        model_path = os.path.join(model_dir, 'model_mobilefacenet.pth')
    elif model_name == 'facenet':
        net = InceptionResnetV1(device=device).to(device)
        model_path = os.path.join(model_dir, 'vggface2.pt')
    else:
        raise Exception('No such model!')
    if train_mode is False:
        net.eval()
    if pretrained:
        net.load_state_dict(torch.load(model_path))
    return net


if __name__ == '__main__':
    net = load_model('resnet50', pretrained=True)
    print(net)