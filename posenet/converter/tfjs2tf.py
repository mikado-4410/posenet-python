import tfjs_graph_converter as tfjs
import os


def convert(model_id, model_dir, check=False):
    tfjsdir = os.path.join('/opt/project/_posenet_weights', 'mobilenet_v1_101')
    tfdir = os.path.join('/opt/project/_models', 'model-mobilenet_v1_101_test')
    tfjs.api.graph_model_to_saved_model(tfjsdir, tfdir, ['serve'])
