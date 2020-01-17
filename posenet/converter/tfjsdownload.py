import urllib.request
import posixpath
import json
import zlib
import os
import shutil
import tensorflowjs.converters.common as tfjs_common
import tfjs_graph_converter.common as tfjs_converter_common
import posenet.converter.common as common

from posenet.converter.config import load_config

TFJS_MODEL_DIR = './_tfjs_models'
TF_MODEL_DIR = './_tf_models'


def model_config(model, neuralnet, model_variant):
    config = load_config()
    tfjs_models = config['models']['tfjs']
    model_cfg = tfjs_models[model][neuralnet]
    return {
        'base_url': model_cfg['base_url'],
        'filename': model_cfg['model_variant'][model_variant]['filename'],
        'output_stride': model_cfg['model_variant'][model_variant]['output_stride'],
        'data_format': model_cfg['model_variant'][model_variant]['data_format'],
        'input_tensors': model_cfg['input_tensors'],
        'output_tensors': model_cfg['output_tensors'],
        'tfjs_dir': os.path.join(TFJS_MODEL_DIR, model, neuralnet, model_variant),
        'tf_dir': os.path.join(TF_MODEL_DIR, model, neuralnet, model_variant)
    }


def fix_model_file(model_cfg):
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])

    if not model_cfg['filename'] == 'model.json':
        # The expected filename for the model json file is 'model.json'.
        # See tfjs_common.ARTIFACT_MODEL_JSON_FILE_NAME in the tensorflowjs codebase.
        normalized_model_json_file = os.path.join(model_cfg['tfjs_dir'], 'model.json')
        shutil.copyfile(model_file_path, normalized_model_json_file)

    with open(model_file_path, 'r') as f:
        json_model_def = json.load(f)

    return json_model_def


def download_single_file(base_url, filename, save_dir):
    output_path = os.path.join(save_dir, filename)
    url = posixpath.join(base_url, filename)
    req = urllib.request.Request(url)
    response = urllib.request.urlopen(req)
    if response.info().get('Content-Encoding') == 'gzip':
        data = zlib.decompress(response.read(), zlib.MAX_WBITS | 32)
    else:
        # this path not tested since gzip encoding default on google server
        # may need additional encoding/text handling if hit in the future
        data = response.read()
    with open(output_path, 'wb') as f:
        f.write(data)


def download_tfjs_model(model, neuralnet, model_variant):
    """
    Download a tfjs model with saved weights.

    :param model: The model, e.g. 'bodypix'
    :param neuralnet: The neural net used, e.g. 'resnet50'
    :param model_variant: The reference to the model file, e.g. 'stride16'
    """
    model_cfg = model_config(model, neuralnet, model_variant)
    model_file_path = os.path.join(model_cfg['tfjs_dir'], model_cfg['filename'])
    if os.path.exists(model_file_path):
        print('Model file already exists: %s...' % model_file_path)
        return
    if not os.path.exists(model_cfg['tfjs_dir']):
        os.makedirs(model_cfg['tfjs_dir'])

    download_single_file(model_cfg['base_url'], model_cfg['filename'], model_cfg['tfjs_dir'])

    json_model_def = fix_model_file(model_cfg)

    shard_paths = json_model_def['weightsManifest'][0]['paths']
    for shard in shard_paths:
        download_single_file(model_cfg['base_url'], shard, model_cfg['tfjs_dir'])
