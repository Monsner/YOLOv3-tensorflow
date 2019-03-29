from tensorflow.python.framework import graph_util
import tensorflow as tf

CHECKPOINT_PATH = 'C:\\Users\\ZhengSL\\Desktop\\TensorRT_yolov3\\YOLOv3_TensorFlow-master\\data\\darknet_weights\\yolov3.ckpt'
PB_PATH = 'C:\\Users\\ZhengSL\\Desktop\\TensorRT_yolov3\\YOLOv3_TensorFlow-master\\data\\darknet_weights\\frozen_inference_graph.pb'

def export_model(input_checkpoint, output_graph):
    # 这个可以加载saver的模型
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, input_checkpoint)
        for op in tf.get_default_graph().get_operations():
            print(op.name) #打印所有节点名称
        # output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        #     sess=sess,
        #     input_graph_def=input_graph_def,  # 等于:sess.graph_def
        #     output_node_names=['yolov3/yolov3_head/feature_map_1','yolov3/yolov3_head/feature_map_2','yolov3/yolov3_head/feature_map_3'])  # 如果有多个输出节点，以逗号隔开这个是重点，输入和输出的参数都需要在这里记录
        #
        # with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        #     f.write(output_graph_def.SerializeToString())  # 序列化输出


export_model(CHECKPOINT_PATH, PB_PATH)