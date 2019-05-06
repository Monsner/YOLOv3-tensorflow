import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import numpy as np
import cv2
from tensorflow.python.platform import gfile

PB_PATH = "./data/darknet_weights/frozen_inference_graph.pb"
TRT_PATH = "./data/darknet_weights/trt_graph.pb"
IMG_PATH = "/media/nvidia/60CD-B390/YOLOv3_Tensorflow-master/data/demo_data/dog.jpg"
IMAGE_SIZE = 160

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)

output_graph_def = tf.GraphDef()
with open(PB_PATH, "rb") as f:
    output_graph_def.ParseFromString(f.read())

trt_graph = trt.create_inference_graph(
    input_graph_def=output_graph_def,
    outputs=["yolov3/yolov3_head/feature_map_1","yolov3/yolov3_head/feature_map_2","yolov3/yolov3_head/feature_map_3"],
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=3)

with open(TRT_PATH, 'wb') as f:
    f.write(trt_graph.SerializeToString())

output_node = tf.import_graph_def(trt_graph, return_elements = ["yolov3/yolov3_head/feature_map_1","yolov3/yolov3_head/feature_map_2","yolov3/yolov3_head/feature_map_3"])

tf_input = tf_sess.graph.get_tensor_by_name("import/Placeholder:0")

img_ori = cv2.imread(IMG_PATH)
height_ori, width_ori = img_ori.shape[:2]
img = cv2.resize(img_ori, tuple([IMAGE_SIZE, IMAGE_SIZE]))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img, np.float32)
img = img[np.newaxis, :] / 255.
img_resized = np.reshape(img, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])


tf_sess.run(output_node, feed_dict={tf_input:img_resized})


# for op in tf.get_default_graph().get_operations():
#     print(op.name)

# summaryWriter = tf.summary.FileWriter('log/', tf_sess.graph)

tf_sess.close()
