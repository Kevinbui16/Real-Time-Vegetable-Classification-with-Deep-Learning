# Vegetable Classification Using Deep Learning

## Introduction

This project has implemented two deep-learning models in classifying vegetables through pictures, videos, and live feeds from cameras: a self-built CNN, and a pre-trained ResNet50 model. It was noted that the ResNet50 model performed much better in accuracy from the first epoch itself at 98%, while the custom-built model took about ten epochs before getting such performance.

## Requirements

To run this project, you will need the following libraries:

| Libraries               |
|------------------------ |
| Pytorch                 |  
| Sklearn                 |  
| OpenCV                  |   
| Numpy                   |  
| argparse                |  
| os                      |  
| shutil                  |  
| Matplotlib              |  

## File Descriptions

- **dataset.py**: Script for loading and preprocessing the dataset.
- **model.py**: Contains the architectures of both the ResNet50 and the custom CNN model.
- **train.py**: Scripts to train ResNet models, with TensorBoard integration for monitoring.
- **train_build_model.py**: Scripts to train CNN models from the file model.py, with TensorBoard integration for monitoring.
- **inference.py, inference_video.py, inference_live_camera.py**: These scripts handle inference for images, videos, and real-time camera feeds, respectively.

## How to use my code
Two folders exist: vegetable_deep_learning_build_model and vegetable_deep_learning_ResNet_model. Both of them have the same structure, except for the models.

### ResNet50 Model

#### - Usage -
* **Run dataset model** by running `python dataset.py`
* **Train Resnet50 model** by running `python train.py` .You can change the parameters inside it.For example: `python train.py -e 30`
* **Test your trained model with image** by running `python inference.py`. For example: `python inference.py -e path/to/image.jpg`
* **Test your trained model with video** by running `python inference_video.py`. For example: `python inference_video.py -e path/to/video.mp4`
* **Test your trained model with live camera** by running `python inference_live_camera.py`

#### -TensorBoard Training Visualizations-

<p align="center">
  <strong><i>-Live Camera Prediction-</i></strong>
</p>

https://github.com/Kevinbui16/Real-Time-Vegetable-Classification-with-Deep-Learning/assets/122188085/3146f2c7-581d-42a5-a90b-e87f6352764f

<p align="center">
  The demo could also be found at: https://youtu.be/JpxL3SVMqXg
</p>


<p align="center">
  <strong><i>Video Prediction</i></strong>
</p>

https://github.com/Kevinbui16/Real-Time-Vegetable-Classification-with-Deep-Learning/assets/122188085/84100d43-fa46-441f-85ca-0966998b31f4



















