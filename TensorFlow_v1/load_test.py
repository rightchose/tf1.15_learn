import numpy as np
import tensorflow as tf
from cnn import CNN
from tensorflow.python.framework import tensor_util

# model restore
# input_size = [32,32,3]
# num_classes = 10
# optimizer = 'Adam'

# model = CNN(input_size=input_size,
#             num_classes=num_classes,
#             optimizer='Adam')

# sess = model.sess
# saver = model.saver
# saver.restore(sess, 'G:/code/tf1.15_learn/TensorFlow_v1/save_dir/test.ckpt')


# frozen pb
pb_file_path  = 'G:/code/tf1.15_learn/TensorFlow_v1/pb_dir_1/test.pb'
tf.reset_default_graph()
print('Loading model...')
graph = tf.Graph()

with tf.gfile.GFile(pb_file_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
print('Check out the input placeholders:')
nodes = [
    n.name + ' => ' + n.op for n in graph_def.node
    if n.op in ('Placeholder')
]

with graph.as_default():
    # Define input tensor
    input = tf.placeholder(np.float32,
                                shape=[None, 32, 32, 3],
                                name='input')
    dropout_rate = tf.placeholder(tf.float32,
                                        shape=[],
                                        name='dropout_rate')
    tf.import_graph_def(graph_def, {
        'input': input,
        'dropout_rate': dropout_rate
    })

graph.finalize()

print('Model loading complete!')

# Get layer names
layers = [op.name for op in graph.get_operations()]
for layer in layers:
    print(layer)

sess = tf.Session(graph=graph)

weight_nodes = [n for n in graph_def.node if n.op == 'Const']
for n in weight_nodes:
    print("Name of the node - %s" % n.name)
    print("Value - " )
    print(tensor_util.MakeNdarray(n.attr['value'].tensor))

print("finish!\n")