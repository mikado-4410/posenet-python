import tensorflow as tf
import os
import posenet.converter.tfjsdownload as tfjsdownload
import posenet.converter.tfjs2tf as tfjs2tf
from posenet.resnet import ResNet
from posenet.mobilenet import MobileNet
from posenet.posenet import PoseNet


def load_model(model, neuralnet, model_variant):

    model_cfg = tfjsdownload.model_config(model, neuralnet, model_variant)
    model_path = model_cfg['tf_dir']
    if not os.path.exists(model_path):
        print('Cannot find tf model path %s, converting from tfjs...' % model_path)
        tfjs2tf.convert(model, neuralnet, model_variant)
        assert os.path.exists(model_path)

    sess = tf.compat.v1.Session()

    sess.graph.as_default()
    tf.compat.v1.saved_model.loader.load(sess, ["serve"], model_path)

    output_tensor_names = model_cfg['output_tensors']
    input_tensor_name = model_cfg['input_tensors']['image']

    if neuralnet == 'resnet50_v1':
        net = ResNet(sess, input_tensor_name, output_tensor_names, model_cfg['output_stride'])
    else:
        net = MobileNet(sess, input_tensor_name, output_tensor_names, model_cfg['output_stride'])

    return PoseNet(net)
