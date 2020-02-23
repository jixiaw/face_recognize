import math
import numpy as np
from PIL import Image
import torch
from .nets import PNet, RNet, ONet
from .utils import nms, calibrate_box, get_image_boxes, convert_to_square, _preprocess, _generate_bboxes, align_crop
from torchvision import transforms


class MTCNN():
    def __init__(self, device=None):
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cpu')
        self.pnet = PNet().to(self.device).eval()
        self.rnet = RNet().to(self.device).eval()
        self.onet = ONet().to(self.device).eval()

    def get_align_faces(self, image, image_size=(112, 112), return_largest=False, return_boxes=False, return_tensor=False):
        bounding_boxes, landmarks = self.detect_faces(image)
        num_faces = len(bounding_boxes)
        if num_faces == 0:
            if return_boxes:
                return [], []
            else:
                return []
        faces = []
        if return_largest:
            face = align_crop(image, landmarks[0])
            face = face.resize(image_size)
            faces.append(face)
            bounding_boxes = bounding_boxes[[0]]
        else:
            for i in range(num_faces):
                face = align_crop(image, landmarks[i])
                face = face.resize(image_size)
                faces.append(face)

        if return_tensor:
            transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            faces = [transf(face) for face in faces]
            faces = torch.stack(faces)


        if return_boxes:
            return faces, bounding_boxes
        else:
            return faces

    def detect_faces(self, image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8],
                     nms_thresholds=[0.7, 0.7, 0.7], return_landmarks=True):
        # pnet, rnet, onet = PNet(), RNet(), ONet()
        # onet.eval()
        width, height = image.size
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        scales = []
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1
        bounding_boxes = []
        for s in scales:  # run P-Net on different scales
            boxes = self.run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
            bounding_boxes.append(boxes)
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = torch.FloatTensor(img_boxes)
        output = self.rnet(img_boxes.to(self.device))
        offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
        probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = torch.FloatTensor(img_boxes)
        output = self.onet(img_boxes.to(self.device))
        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
        probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        if return_landmarks:
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

            areas = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1])
            sort_index = np.argsort(areas)[::-1]
            bounding_boxes = bounding_boxes[sort_index]
            landmarks = landmarks[sort_index]

            return bounding_boxes, landmarks
        else:
            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]

            areas = (bounding_boxes[:, 2] - bounding_boxes[:, 0]) * (bounding_boxes[:, 3] - bounding_boxes[:, 1])
            sort_index = np.argsort(areas)[::-1]
            bounding_boxes = bounding_boxes[sort_index]

            return bounding_boxes

    def run_first_stage(self, image, net, scale, threshold):
        """
            Run P-Net, generate bounding boxes, and do NMS.
        """
        width, height = image.size
        sw, sh = math.ceil(width * scale), math.ceil(height * scale)
        img = image.resize((sw, sh), Image.BILINEAR)
        img = np.asarray(img, 'float32')
        img = torch.FloatTensor(_preprocess(img))

        output = net(img.to(self.device))
        probs = output[1].cpu().data.numpy()[0, 1, :, :]
        offsets = output[0].cpu().data.numpy()

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        return boxes[keep]