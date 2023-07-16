from pathlib import Path
from cnn import CNN

import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants


input_size = [32,32,3]
num_classes = 10
optimizer = 'Adam'

# build model
model = CNN(input_size=input_size, num_classes=num_classes, optimizer=optimizer)

sess = model.sess
saver = model.saver
output = model.output

# tf.train.Saver

# save => checkpoint, test.ckpt.data-00000-of-00001, test.ckpt.index, test.ckpt.meta
saver.save(sess, "G:/code/tf1.15_learn/TensorFlow_v1/save_dir/test.ckpt")

# save signature
# save => saved_model.pb, variables {variables.data-00000-of-00001, variables.index}
input = tf.placeholder(tf.float32, [None] + input_size, name='input')
label = tf.placeholder(tf.float32, [None, num_classes], name='label')
dropout_rate = tf.placeholder(tf.float32, shape=[], name='dropout_rate')
signature = signature_def_utils.build_signature_def(
    inputs={
        'input':
        saved_model_utils.build_tensor_info(input),
        'dropout_rate':
        saved_model_utils.build_tensor_info(dropout_rate)
    },
    outputs={
            'output': saved_model_utils.build_tensor_info(output)
        },
    method_name=signature_constants.PREDICT_METHOD_NAME)

signature_map = {
    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
}
model_builder = saved_model_builder.SavedModelBuilder('G:/code/tf1.15_learn/TensorFlow_v1/save_signature')
model_builder.add_meta_graph_and_variables(
    sess,
    tags=[tag_constants.SERVING],
    signature_def_map=signature_map,
    clear_devices=True)
model_builder.save(as_text=False)

# save as pb
pb_dir = 'G:/code/tf1.15_learn/TensorFlow_v1/pb_dir_1'
pbtxt_filename = os.path.join(pb_dir, 'test.pbtxt')
pb_filename = os.path.join(pb_dir, 'test.pb')
tf.train.write_graph(graph_or_graph_def=sess.graph_def, 
                        logdir='G:/code/tf1.15_learn/TensorFlow_v1/log_dir',
                        name=pbtxt_filename,
                        as_text=True)

# way1 freeze graph
freeze_graph.freeze_graph(input_graph=pbtxt_filename,
                            input_saver='',
                            input_binary=False,
                            input_checkpoint='G:/code/tf1.15_learn/TensorFlow_v1/save_dir/test.ckpt', # 先前save的ckpt目录
                            output_node_names='cnn/output',
                            restore_op_name='save/restore_all',
                            filename_tensor_name='save/Const:0',
                            output_graph=pb_filename,
                            clear_devices=True,
                            initializer_nodes='')

# way2 no freeze
pb_dir = 'G:/code/tf1.15_learn/TensorFlow_v1/pb_dir_2'
if not os.path.exists(pb_dir):
        os.makedirs(pb_dir)
pb_filename = os.path.join(pb_dir, 'test.pb')
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()
output_node_names = ['cnn/output']
output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)
with tf.gfile.GFile(pb_filename, 'wb') as f:
    f.write(output_graph_def.SerializeToString())

print("finish!\n")
