#  YOLOv3_TensorFlow

### 0. 运行方法

主要借鉴了项目https://github.com/wizyoung/YOLOv3_TensorFlow，为了在TX2上使用TF-TRT加速进行了一些改动。

TF-TRT的使用主要参考了https://github.com/NVIDIA-AI-IOT/tf_trt_models，以及TF-TRT官方文档https://docs.nvidia.com/deeplearning/dgx/tf-trt-user-guide/index.html。

我使用的运行环境：

tensorflow1.8 - gpu 

python2.7 (理论上python3也可以运行，可能需要稍微修改一下代码)

#### 运行过程：

- 首先下载YOLOv3的权重文件，下载地址[here](https://pjreddie.com/media/files/yolov3.weights). 把 weights 文件放在这个目录下 `./data/darknet_weights/` 然后运行:

```shell
python convert_weight.py
```

注意转换时需要设置输入图片大小image_size（32的倍数，建议先使用160尝试）。运行后将会生成TensorFlow checkpoint模型，存储在这个目录： `./data/darknet_weights/` 。

- 然后需要把checkpoint模型转化为固化的pb模型，即frozen_graph。运行：

~~~shell
python export_pb.py
~~~

得到frozen_graph_def.pb模型，也存储在 `./data/darknet_weights/`目录下。

你可以运行yolov3pb.py来查看是否可以正常运行pb模型进行推理（需要设置IMAGE_SIZE），在这里如果图片较大的话可能会被killed，遇到这个问题可以尝试改小网络的输入图片大小。

- 接下来我们需要把pb模型使用TensorRT加速，使用TF-TRT API接口，运行：

```shell 
python pb_to_trt.py
```

该步骤也需要在pb_to_trt.py中设置IMAGE_SIZE的大小。运行后得到构建的引擎图文件trt_graph.pb，仍存储在 `./data/darknet_weights/`目录下。

如果你要更改TensorRT加速时的相关设置，只需要更改trt.create_inference_graph()函数中的参数：

```shell
trt_graph = trt.create_inference_graph(
    input_graph_def=output_graph_def,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=5
)
```

具体的参数含义参考TF-TRT文档https://docs.nvidia.com/deeplearning/dgx/tf-trt-user-guide/index.html。

- 最后，使用加速后的pb模型进行推理（同样，不要忘记在python文件中设置IMAGE_SIZE以及测试图片路径等参数），运行：

```shell
python trt.py
```

执行推理时，我遇到的报错如下：

```shell
2019-04-29 04:31:34.738939: F tensorflow/contrib/tensorrt/shape_fn/trt_shfn.cc:52] TensorRT engine requires consistent batch size
Aborted(core dumped)
```

该报错仍未解决。

#### 其他

- ckpt文件的执行可以使用test_single_image.py文件以及video_test.py，你需要指定图片或视频文件的路径。
- pb_to_tensorboard.py用于使用tensorboard查看pb文件的网络结构，tensorrt构建的trt_graph.pb也可以使用tensorboard查看.
- yolov3_trt.py相当于pb_to_trt.py与trt.py的集合，将pb文件转化为trt模型并直接执行推理，不过该文件测试比较少可能有一些bug。



### 以下是原项目的readme

### 1. Introduction

This is my implementation of [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) in pure TensorFlow. It contains the full pipeline of training and evaluation on your own dataset. The key features of this repo are:

- Efficient tf.data pipeline
- Weights converter (converting pretrained darknet weights on COCO dataset to TensorFlow checkpoint.)
- Extremely fast GPU non maximum supression.
- Full training pipeline.
- Kmeans algorithm to select prior anchor boxes.
- [ ] Multi-GPU training with sync batch norm. (on working)

### 2. Requirements

- tensorflow >= 1.8.0 (lower versions may work too)
- opencv-python

### 3. Weights convertion

The pretrained darknet weights file can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). Place this weights file under directory `./data/darknet_weights/` and then run:

```shell
python convert_weight.py
```

Then the converted TensorFlow checkpoint file will be saved to `./data/darknet_weights/` directory.

You can also download the converted TensorFlow checkpoint file by me via [[Google Drive link](https://drive.google.com/drive/folders/1mXbNgNxyXPi7JNsnBaxEv1-nWr7SVoQt?usp=sharing)] or [[Github Release](https://github.com/wizyoung/YOLOv3_TensorFlow/releases/)]and then place it to the same directory.

### 4. Running demos

There are some demo images and videos under the `./data/demo_data/`. You can run the demo by:

Single image test demo:

```shell
python test_single_image.py ./data/demo_data/messi.jpg
```

Video test demo:

```shell
python video_test.py ./data/demo_data/video.mp4
```

Some results:

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/dog.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/messi.jpg?raw=true)

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/data/demo_data/results/kite.jpg?raw=true)

Compare the kite detection results with TensorFlow's offical API result [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/img/kites_detections_output.jpg).

(The kite detection result is under input image resolution 1344x896)

### 5. Inference speed

How fast is the inference speed? With images scaled to 416*416:


| Backbone              |   GPU    | Time(ms) |
| :-------------------- | :------: | :------: |
| Darknet-53 (paper)    | Titan X  |    29    |
| Darknet-53 (my impl.) | Titan XP |   ~23    |

why is it so fast? Check the ImageNet classification result comparision from the paper:

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/docs/backbone.png?raw=true)

### 6. Model architecture

For better understanding of the model architecture, you can refer to the following picture. With great thanks to [Levio](https://blog.csdn.net/leviopku/article/details/82660381) for your excellent work!

![](https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/docs/yolo_v3_architecture.png?raw=true)

### 7. Training

#### 7.1 Data preparation 

(1) annotation file

Generate `train.txt/val.txt/test.txt` files under `./data/my_data/` directory. One line for one image, in the format like `image_absolute_path box_1 box_2 ... box_n`. Box_format: `label_index x_min y_min x_max y_max`.(The origin of coordinates is at the left top corner.)

For example:

```
xxx/xxx/1.jpg 0 453 369 473 391 1 588 245 608 268
xxx/xxx/2.jpg 1 466 403 485 422 2 793 300 809 320
...
```

**NOTE**: **You should leave a blank line at the end of each txt file.**

(2)  class_names file:

Generate the `data.names` file under `./data/my_data/` directory. Each line represents a class name.

For example:

```
bird
person
bike
...
```

The COCO dataset class names file is placed at `./data/coco.names`.

(3) prior anchor file:

Using the kmeans algorithm to get the prior anchors:

```
python get_kmeans.py
```

Then you will get 9 anchors and the average IOU. Save the anchors to a txt file.

The COCO dataset anchors offered by YOLO v3 author is placed at `./data/yolo_anchors.txt`, you can use that one too.

**NOTE: The yolo anchors should be scaled to the rescaled new image size. Suppose your image size is [W, H], and the image will be rescale to 416*416 as input, for each generated anchor [anchor_w, anchor_h], you should apply the transformation anchor_w = anchor_w / W * 416, anchor_h = anchor_g / H * 416.**

#### 7.2 Training

Using `train.py`. The parameters are as following:

```shell
$ python train.py -h
usage: train.py [-h] [--train_file TRAIN_FILE] [--val_file VAL_FILE]
                [--restore_path RESTORE_PATH] 
                [--save_dir SAVE_DIR]
                [--log_dir LOG_DIR] 
                [--progress_log_path PROGRESS_LOG_PATH]
                [--anchor_path ANCHOR_PATH]
                [--class_name_path CLASS_NAME_PATH] [--batch_size BATCH_SIZE]
                [--img_size [IMG_SIZE [IMG_SIZE ...]]]
                [--total_epoches TOTAL_EPOCHES]
                [--train_evaluation_freq TRAIN_EVALUATION_FREQ]
                [--val_evaluation_freq VAL_EVALUATION_FREQ]
                [--save_freq SAVE_FREQ] [--num_threads NUM_THREADS]
                [--prefetech_buffer PREFETECH_BUFFER]
                [--optimizer_name OPTIMIZER_NAME]
                [--save_optimizer SAVE_OPTIMIZER]
                [--learning_rate_init LEARNING_RATE_INIT] [--lr_type LR_TYPE]
                [--lr_decay_freq LR_DECAY_FREQ]
                [--lr_decay_factor LR_DECAY_FACTOR]
                [--lr_lower_bound LR_LOWER_BOUND]
                [--restore_part [RESTORE_PART [RESTORE_PART ...]]]
                [--update_part [UPDATE_PART [UPDATE_PART ...]]]
                [--update_part [UPDATE_PART [UPDATE_PART ...]]]
                [--use_warm_up USE_WARM_UP] [--warm_up_lr WARM_UP_LR]
                [--warm_up_epoch WARM_UP_EPOCH]
```

Check the `train.py` for more details. You should set the parameters yourself. 

Some training tricks in my experiment:

(1) Apply the two-stage training strategy:

First stage: Restore `darknet53_body` part weights from COCO checkpoints, train the `yolov3_head` with big learning rate like 1e-3 until the loss reaches to a low level, like less than 1.

Second stage: Restore the weights from the first stage, then train the whole model with small learning rate like 1e-4 or smaller. At this stage remember to restore the optimizer parameters if you use optimizers like adam.

(2) Quick train:

If you want to obtain good results in a short time like in 10 minutes. You can use the coco names but substitute several with real class names in your dataset. In this way you restore the whole pretrained COCO model and get a 80 class classification model, but you only care the class names from your dataset.

### 8. Evaluation

Using `eval.py` to evaluate the validation or test dataset. The parameters are as following:

```shell
$ python eval.py -h
usage: eval.py [-h] [--eval_file EVAL_FILE] [--restore_path RESTORE_PATH]
               [--anchor_path ANCHOR_PATH] 
               [--class_name_path CLASS_NAME_PATH]
               [--batch_size BATCH_SIZE]
               [--img_size [IMG_SIZE [IMG_SIZE ...]]]
               [--num_threads NUM_THREADS]
               [--prefetech_buffer PREFETECH_BUFFER]
```

Check the `eval.py` for more details. You should set the parameters yourself. 

You will get the loss, recall and precision metrics results, like:

```shell
recall: 0.927, precision: 0.945
total_loss: 0.210, loss_xy: 0.010, loss_wh: 0.025, loss_conf: 0.125, loss_class: 0.050
```

### 9. Other tricks

There are many skills you can try during training:

(1) Data augmentation: You can implement your data augmentation like color jittering under `data_augmentation` method in `./utils/data_utils.py`.

(2) Mixed up and label smoothing like what [Gluon-CV](https://github.com/dmlc/gluon-cv/tree/master/scripts/detection/yolo) does.

(3) Normalizations like L2 norm.

(4) Mutil-scale training: You can change the input image scales (i.e. different input resolutions) periodically like the author does in the original paper.

-------

### Credits:

folk from https://github.com/wizyoung/YOLOv3_TensorFlow







 