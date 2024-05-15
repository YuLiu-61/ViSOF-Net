# 1. Introduction
We propose a novel ViSOF-Net that exhibits state-of-the-art performance. It introduces recurrent optical flow estimation into object detection for safety monitoring. Our ViSOF-Net achieves 89.5% AP``^{[.5:.05:.95]}``, 85.3% AP``^{S}``, 88.9% AP``^{M}``, 94.7% AP``^{L}`` on our proposed large scale dataset PPE725 vid. It outperforms all the detection models including YOLOv9, which was proposed a few months ago.

# 2.  Performance

| Model         | Input Size | Parameters(M) | AP(.5:.0.5:.95) | AP(S) | AP(M) | AP(L) |
|---------------|:-----------|---------------|-----------------|:------|:------|:------|
| YOLOv7        | 640        | 37            | 88.3            |<u> 85.1</u>  | 86.2  | 94.1  |
| YOLOv8        | 640        | 44            | 85.9            | 79.9  | 83.6  | 93.1  |
| YOLOv9        | 640        | 69            | <u>88.8 </u>           | 83.1  | <u>87.3</u>  | **94.8**  |
| RT-DETR       | 640        | 42            | 79.4            | 75.2  | 75.2  | 88.6  |
| PRB-FPN-MSP   | 640        | 106           | 85.3            | 81.8  | 84.3  | 89.8  |
| Cascade R-CNN | [800,1333] | 272           | 77.1            | 73    | 75.1  | 83.4  |
| **ViSOF-Net**     | 640        | 41            | **89.5**            | **85.3**  | **88.9**  | <u>94.7</u>  |

To get the weights of models in the table：
Baidu Cloud Link: https://pan.baidu.com/s/1xi4qsk5ILysNitc0SnHOQA 
Code: wbvo

# 3. Install
```github
git clone https://github.com/YuLiu-61/ViSOF-Net.git
cd ViSOF-Net
pip install -r requirements.txt
```
# 4. Training
## Data preparation
1. Download hat dataset images and labels from:
Baidu Cloud Link: https://pan.baidu.com/s/1jT4Yyhq2f_86V48gmioW0g 
Code: oe9r
This dataset includes 12,277 images of 2 classes(hat and person).
If training on custom dataset, make sure your dataset structure is YOLO format.

2. Download yolov7.pt from following link and put it in /weight directory.
Baidu Cloud Link: https://pan.baidu.com/s/1AXFxbt7Eu8IcHkexPTk2pA 
Code: bcoo

## CFG preparation
1.Modify the dataset information in data/coco.yaml.
2.Modify cfg/training/yolov7-temporal.yaml, change nc as your dataset’s number of classes.

## Single GPU training 
Our model only support one-batch training now, you can start training as follows:
python train.py --cfg cfg/training/yolov7-temporal.yaml --data data/yourDataset.yaml

# 5. Test
## Data preparation
Download hat dataset video frames and labels from :
Baidu Cloud Link：https://pan.baidu.com/s/19l5ds5_TyEOCdzjBY92ZMQ 
Code：gr69
This dataset includes 170,974 video frames of 2 classes(hat and person).

## CFG preparation
1. Modify the val path with your test dataset path in data/coco.yaml.
2. Modify class Temporal in models/common.py by replacing #temporal part for train with # 17 frames aggregation for detect and test.

## Single GPU training 
You can start testing as follows:
```github
python test.py --weights {weights of your own model} --data data/yourDataset.yaml
```

# 6. Inference
## On video
```github
python detect.py --weights {weights of your own model} --source {path of video}
```
### On image
```github
python detect.py --weights {weights of your own model} --source {path of image}
```
