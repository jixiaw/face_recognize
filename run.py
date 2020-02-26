from flask import make_response, url_for, redirect, Flask, request, render_template, session, Response
import time
import cv2
from detect_pipeline import FaceRecognizePipeline
import torch


app = Flask(__name__)
device = torch.device('cuda:0')
pipeline = FaceRecognizePipeline(device=device)


class VideoCamera(object):
    def __init__(self):
        # 通过 opencv 获取实时视频流
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        assert success
        image = pipeline.forward(image, detect_only=False)
        ret, jpeg = cv2.imencode('.jpg', image)
        assert ret
        return jpeg.tobytes()


@app.route('/')  # 主页
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        # 使用 generator 函数输出视频流， 每次请求输出的 content 类型是 image/jpeg
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')  # 这个地址返回视频流响应
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)