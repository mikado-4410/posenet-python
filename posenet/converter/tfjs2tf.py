import os
import tensorflow as tf
import tfjs_graph_converter as tfjs
import posenet.converter.tfjsdownload as tfjsdownload


def convert(model, neuralnet, model_variant):
    model_cfg = tfjsdownload.model_config(model, neuralnet, model_variant)
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])
    if not os.path.exists(model_file_path):
        print('Cannot find tfjs model path %s, downloading tfjs model...' % model_file_path)
        tfjsdownload.download_tfjs_model(model, neuralnet, model_variant)
    tfjs.api.graph_model_to_saved_model(model_cfg['tfjs_dir'], model_cfg['tf_dir'], ['serve'])


def list_tensors(model, neuralnet, model_variant):
    model_cfg = tfjsdownload.model_config(model, neuralnet, model_variant)
    graph = tfjs.api.load_graph_model(model_cfg['tfjs_dir'])
    with tf.compat.v1.Session(graph=graph) as sess:
        # the module provides some helpers for querying model properties
        input_tensor_names = tfjs.util.get_input_tensors(graph)
        output_tensor_names = tfjs.util.get_output_tensors(graph)

        print('input tensors:')
        for it in input_tensor_names:
            print(it)
        print('--')
        print('output tensors:')
        for ot in output_tensor_names:
            print(ot)
