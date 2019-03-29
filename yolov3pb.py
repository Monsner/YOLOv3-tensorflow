import tensorflow as tf
import  numpy as np
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

def recognize(jpg_path, pb_file_path):
    anchors = parse_anchors("./data/yolo_anchors.txt")
    classes = read_class_names("./data/coco.names")
    num_class = len(classes)

    color_table = get_color_table(num_class)


    img_ori = cv2.imread(jpg_path)
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple([416, 416]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("Placeholder:0")
            print(input_x)
            feature_map_1 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_1:0")
            print(feature_map_1)
            feature_map_2 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_2:0")
            print(feature_map_2)
            feature_map_3 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_3:0")
            print(feature_map_3)
            features = feature_map_1, feature_map_2, feature_map_3

            # img = io.imread(jpg_path)
            # img = transform.resize(img, (416, 416, 3))
            yolo_model = yolov3(num_class, anchors)
            input_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='input_data')
            yolo_model.pb_forward(input_data)
            features = sess.run(features, feed_dict={input_x:np.reshape(img, [-1, 416, 416, 3])})
            feature1, feature2, feature3 = features
            feature1 = tf.convert_to_tensor(feature1)
            feature2 = tf.convert_to_tensor(feature2)
            feature3 = tf.convert_to_tensor(feature3)
            features = feature1, feature2, feature3

            pred_boxes, pred_confs, pred_probs = yolo_model.predict(features)

            pred_scores = pred_confs * pred_probs

            boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.4,
                                            iou_thresh=0.5)


            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

            # rescale the coordinates to the original image
            boxes_[:, 0] *= (width_ori / float(416))
            boxes_[:, 2] *= (width_ori / float(416))
            boxes_[:, 1] *= (height_ori / float(416))
            boxes_[:, 3] *= (height_ori / float(416))

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
            cv2.imshow('Detection result', img_ori)
            cv2.imwrite('detection_result.jpg', img_ori)
            cv2.waitKey(0)

            #
            # print("img_out",img_out_softmax)
            # prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print("label:",prediction_labels)

recognize("./data/demo_data/kite.jpg", "./data/darknet_weights/frozen_inference_graph.pb")