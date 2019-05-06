import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import  numpy as np
import cv2
import time

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

IMAGE_SIZE = 160
IMG_PATH = "./data/demo_data/cat.jpg"
TRT_PATH = "./data/darknet_weights/trt_graph.pb"

def recognize(jpg_path, pb_file_path):
    anchors = parse_anchors("./data/yolo_anchors.txt")
    classes = read_class_names("./data/coco.names")
    num_class = len(classes)

    color_table = get_color_table(num_class)


    img_ori = cv2.imread(jpg_path)
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple([IMAGE_SIZE, IMAGE_SIZE]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    # img = img[np.newaxis, :] / 255.    
    # img_resized = np.reshape(img, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    with tf.Graph().as_default():
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config = tf_config) as sess:
            print "Load TRT_Graph File ..."
            with open(pb_file_path, "rb") as f:
                output_graph_def = tf.GraphDef()
                output_graph_def.ParseFromString(f.read())
            print "Finished"
            input_name = "import/Placeholder"
            output_name1 = "import/yolov3/yolov3_head/feature_map_1"
            output_name2 = "import/yolov3/yolov3_head/feature_map_2"
            output_name3 = "import/yolov3/yolov3_head/feature_map_3"
            output_names = [output_name1, output_name2, output_name3]

            yolo_model = yolov3(num_class, anchors)

            print "Import TRT Graph ..."
            output_node = tf.import_graph_def(output_graph_def, return_elements = ["yolov3/yolov3_head/feature_map_1","yolov3/yolov3_head/feature_map_2","yolov3/yolov3_head/feature_map_3"])
            print "Finished"
            # for op in tf.get_default_graph().as_graph_def().node:
            #    print(op.name)
            
            tf_input = sess.graph.get_tensor_by_name(input_name + ':0')
            feature_map_1 = sess.graph.get_tensor_by_name(output_name1 + ":0")
            feature_map_2 = sess.graph.get_tensor_by_name(output_name2 + ":0")
            feature_map_3 = sess.graph.get_tensor_by_name(output_name3 + ":0")
            features = feature_map_1, feature_map_2, feature_map_3
            sess.run(output_node, feed_dict={tf_input:img[None, ...]})
            print "1111111"

            yolo_model.pb_forward(tf_input)

            pred_boxes, pred_confs, pred_probs = yolo_model.predict(features)

            pred_scores = pred_confs * pred_probs
            print "Detection ......"
            boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.4,
                                            iou_thresh=0.5)
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={tf_input:img[None, ...]})

            # rescale the coordinates to the original image
            boxes_[:, 0] *= (width_ori / float(IMAGE_SIZE))
            boxes_[:, 2] *= (width_ori / float(IMAGE_SIZE))
            boxes_[:, 1] *= (height_ori / float(IMAGE_SIZE))
            boxes_[:, 3] *= (height_ori / float(IMAGE_SIZE))

            print("box coords:")
            print(boxes_)
            print('*' * 30)
            print("scores:")
            print(scores_)
            print('*' * 30)
            print("labels:")
            print(labels_)
            
            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
            # cv2.imshow('Detection result', img_ori)
            cv2.imwrite('detection_result.jpg', img_ori)
            # cv2.waitKey(0)

            #
            # print("img_out",img_out_softmax)
            # prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print("label:",prediction_labels)

recognize(IMG_PATH,TRT_PATH)
