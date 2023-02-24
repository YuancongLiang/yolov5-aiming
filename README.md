## Introduction

CSGO self-aiming program based on YoloV5

## Statement

This project can only be used for learning and communication, not for commercial use, and cannot be used for illegal purposes (including but not limited to: for making game plug-ins, etc.).

## How to use

### Train

This project does not come with a training model. Please go to YoloV5 official website to view related documents.

[ultralytics/yolov5: YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/yolov5)

I provided the model I trained, but the amount of data is not large. If you want better results, please train a new model yourself.

Even so, after the environment is configured, you can still use the project directly.

## Environment configuration

Clone repo and install requirements.txt in a [**Python>=3.8**](https://www.python.org/) environment, including [**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```
git clone https://github.com/DidUSeeMyElk/yolov5-aiming
cd yolov5-aiming
pip install -r requirements.txt
```

Because my screen resolution is 1920*1080, all parameters are set according to this resolution.

If your screen resolution is different from mine, please modify it yourself.

After completing these configurations, you can run this code. However, he may be unresponsive and report no errors.

It could be that you have mouse raw data input turned on in game. Just turn it off.