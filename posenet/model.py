import tensorflow as tf
import os
import posenet.converter.tfjsdownload as tfjsdownload
import posenet.converter.tfjs2tf as tfjs2tf


def load_tf_model(sess, model, neuralnet, model_variant):
    model_cfg = tfjsdownload.model_config(model, neuralnet, model_variant)
    model_path = model_cfg['tf_dir']
    if not os.path.exists(model_path):
        print('Cannot find tf model path %s, converting from tfjs...' % model_path)
        tfjs2tf.convert(model, neuralnet, model_variant)
        assert os.path.exists(model_path)

    sess.graph.as_default()
    tf.compat.v1.saved_model.loader.load(sess, ["serve"], model_path)

    output_tensor_map = model_cfg['output_tensors']

    output_tensors = [
        tf.sigmoid(sess.graph.get_tensor_by_name(output_tensor_map['heatmap']), 'heatmap'),
        sess.graph.get_tensor_by_name(output_tensor_map['offsets']),
        sess.graph.get_tensor_by_name(output_tensor_map['displacement_fwd']),
        sess.graph.get_tensor_by_name(output_tensor_map['displacement_bwd'])
    ]

    return model_cfg['output_stride'], output_tensors
