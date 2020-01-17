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

    loaded_model = tf.saved_model.load(model_path)

    signature_key = tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    print('We use the signature key %s It should be in the keys list:' % signature_key)
    for sig in loaded_model.signatures.keys():
        print('signature key: %s' % sig)

    model_function = loaded_model.signatures[signature_key]
    print('model outputs: %s' % model_function.structured_outputs)

    output_tensor_names = model_cfg['output_tensors']
    output_stride = model_cfg['output_stride']

    if neuralnet == 'resnet50_v1':
        net = ResNet(model_function, output_tensor_names, output_stride)
    else:
        net = MobileNet(model_function, output_tensor_names, output_stride)

    return PoseNet(net)
