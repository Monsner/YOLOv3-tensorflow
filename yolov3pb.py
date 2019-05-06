import tensorflow as tf
import  numpy as np
import cv2
import time

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box

from model import yolov3

IMAGE_SIZE = 160
IMG_PATH = "./data/demo_data/dog.jpg"
PB_PATH = "./data/darknet_weights/frozen_inference_graph.pb"

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
    img = img[np.newaxis, :] / 255.
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        print ("Load Frozen_Graph File ...")
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
        print ("Finished")

        # GPU_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        config = tf.ConfigProto()# gpu_options=GPU_options)
        config.gpu_options.allow_growth = True

        with tf.Session(config = config) as sess:

            # Define Input and Outputs
            input_x = sess.graph.get_tensor_by_name("Placeholder:0")
            feature_map_1 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_1:0")
            feature_map_2 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_2:0")
            feature_map_3 = sess.graph.get_tensor_by_name("yolov3/yolov3_head/feature_map_3:0")
            features = feature_map_1, feature_map_2, feature_map_3
            # yolo config
            yolo_model = yolov3(num_class, anchors)
            yolo_model.pb_forward(input_x)
            # # use frozen_graph to inference
            # print "RUN Graph ..."
            # features = sess.run(features, feed_dict={input_x:np.reshape(img, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])})
            # print "Finished"
            # feature1, feature2, feature3 = features

            # feature1 = tf.convert_to_tensor(feature1)
            # feature2 = tf.convert_to_tensor(feature2)
            # feature3 = tf.convert_to_tensor(feature3)
            # features = feature1, feature2, feature3
            print "Predicting ..."

            pred_boxes, pred_confs, pred_probs = yolo_model.predict(features)
            pred_scores = pred_confs * pred_probs

            boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=30, score_thresh=0.4,
                                            iou_thresh=0.5)
            t0 = time.time()
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_x:np.reshape(img, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])})
            t1 = time.time()
            print "Finished"

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
            print("runtime:")
            print(t1-t0)

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]], color=color_table[labels_[i]])
            #cv2.imshow('Detection result', img_ori)
            cv2.imwrite('pb_result.jpg', img_ori)
            #cv2.waitKey(0)
            num_samples = 50

            t0 = time.time()
            for i in range(num_samples):
                boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_x:np.reshape(img, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])})
            t1 = time.time()
            print('Average runtime: %f seconds' % (float(t1 - t0) / num_samples))

            #
            # print("img_out",img_out_softmax)
            # prediction_labels = np.argmax(img_out_softmax, axis=1)
            # print("label:",prediction_labels)

recognize(IMG_PATH, PB_PATH)
