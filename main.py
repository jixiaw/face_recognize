from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import cv2
from detect_pipeline import FaceRecognizePipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

from MainWindow import Ui_MainWindow


class FaceRecognizeWindow(Ui_MainWindow, QMainWindow):
    def __init__(self, parent=None):
        super(FaceRecognizeWindow, self).__init__(parent)
        self.setupUi(self)

        self.face_recognize = FaceRecognizePipeline(device=torch.device('cuda:0'))
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.keep_recognizing = False  # 是否在识别
        self.keep_detecting = False    # 是否在检测
        self.open_camera = False       # 摄像头是否开启
        self.cap = None
        self.img = None
        self.video_img = None

        # 槽函数
        self.timer_camera.timeout.connect(self.show_camera)
        self.startRecognize.clicked.connect(self.change_recognize_model)
        self.startDetect.clicked.connect(self.change_detect_model)
        self.openCamera.clicked.connect(self.change_camera_model)
        self.selectPhoto.clicked.connect(self.open_photo)

    def open_photo(self):
        if self.open_camera:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请先关闭摄像头!",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.img, imgType = QFileDialog.getOpenFileName(self, '选择图片', '/', 'Image files(*.jpg *.gif *.png)')
            jpg = QtGui.QPixmap(self.img).scaled(self.videoLabel.width(), self.videoLabel.height())
            self.videoLabel.setPixmap(jpg)
            print(self.img)

    def show_camera(self):
        flag, self.video_img = self.cap.read()
        if (flag):
            show = cv2.resize(self.video_img, (self.videoLabel.width(), self.videoLabel.height()))
            if self.keep_recognizing:
                show = self.face_recognize.forward(show, False)
            elif self.keep_detecting:
                show = self.face_recognize.forward(show, True)
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.videoLabel.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage

    def change_recognize_model(self):
        if self.open_camera is False and self.img is None:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请打开摄像头或选择照片!",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        elif self.open_camera:
            if self.keep_recognizing is False:
                self.keep_recognizing = True
                self.keep_detecting = True
                self.startRecognize.setText("停止识别")
                self.startDetect.setText("停止检测")
            else:
                self.keep_recognizing = False
                self.startRecognize.setText("开始识别")
        elif self.img is not None:
            print(self.img)
            img_plt = Image.open(self.img).resize((self.videoLabel.width(), self.videoLabel.height()))
            recognized_img = self.face_recognize.forward(img_plt)
            recognized_img = cv2.cvtColor(recognized_img, cv2.COLOR_BGR2RGB)
            recognized_img = QtGui.QImage(recognized_img.data, recognized_img.shape[1], recognized_img.shape[0],
                                      QtGui.QImage.Format_RGB888)
            self.videoLabel.setPixmap(QtGui.QPixmap.fromImage(recognized_img))

    def change_detect_model(self):
        if self.open_camera is False and self.img is None:
            msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请打开摄像头或选择照片!",
                                                buttons=QtWidgets.QMessageBox.Ok,
                                                defaultButton=QtWidgets.QMessageBox.Ok)
        elif self.open_camera:
            if self.keep_detecting is False:
                self.keep_detecting = True
                self.startDetect.setText("停止检测")
            else:
                self.keep_detecting = False
                self.keep_recognizing = False
                self.startDetect.setText("开始检测")
                self.startRecognize.setText("开始识别")
        elif self.img is not None:
            print(self.img)
            img_plt = Image.open(self.img).resize((self.videoLabel.width(), self.videoLabel.height()))
            recognized_img = self.face_recognize.forward(img_plt, True)
            recognized_img = cv2.cvtColor(recognized_img, cv2.COLOR_BGR2RGB)
            recognized_img = QtGui.QImage(recognized_img.data, recognized_img.shape[1], recognized_img.shape[0],
                                          QtGui.QImage.Format_RGB888)
            self.videoLabel.setPixmap(QtGui.QPixmap.fromImage(recognized_img))

    def change_camera_model(self):
        if self.open_camera is False:
            self.open_camera = True
            self.img = None
            self.openCamera.setText("关闭摄像头")

            self.cap = cv2.VideoCapture(0)
            self.timer_camera.start(30)  # 定时器开始计时30ms，每过30ms从摄像头中取一帧显示
        else:
            self.open_camera = False
            self.keep_detecting = False
            self.keep_recognizing = False
            self.timer_camera.stop()
            self.cap.release()
            self.videoLabel.setText(u"无图像")
            self.startDetect.setText("开始检测")
            self.startRecognize.setText("开始识别")
            self.openCamera.setText("开启摄像头")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = FaceRecognizeWindow()              # 实例化
    ui.show()                               # 调用 ui.show() 来显示。show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())                   # 不加这句，程序界面会一闪而过
