import json
import struct
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph
import cv2
import numpy as np
import os
import tempfile

from posenet.converter.config import load_config

# BASE_DIR = os.path.join(tempfile.gettempdir(), '_posenet_weights')
BASE_DIR = os.path.join('.', '_posenet_weights')

# Note that this file contains reverse-engineered documentation that contains several notes about points that need to be verified.


def to_output_strided_layers(convolution_def, output_stride):
    """
    There seem to be some magic formulas used in this function. The output magically aligns with the details of the layer definition
    for MobilenetV1. Not sure how reusable this is for other networks that use depthwise convolutions.

    Note: Verify whether we can reuse this function for other networks, like MobilenetV2.

    :param convolution_def: A MobileNet convolution definition selection from the config.yaml file.
    :param output_stride: The chosen output stride. Note to check how the output stride is coupled to the chosen network
    variables (see the load_variables function).
    :return: An array containing an element for each layer with the detailed layer specs defined in each of them.
    """

    current_stride = 1
    rate = 1
    block_id = 0
    buff = []
    for _a in convolution_def:
        conv_type = _a[0]
        stride = _a[1]
        
        if current_stride == output_stride: # How often do we get here?
            layer_stride = 1  # tf.nn.depthwise_conv2d nets require the strides to be 1 when the rate (dilation) is >1
            layer_rate = rate
            rate *= stride    # why is this?
        else:
            layer_stride = stride
            layer_rate = 1    # tf.nn.depthwise_conv2d nets require the rate (dilation) to be 1 when the strides are >1
            current_stride *= stride # why is this?
        
        buff.append({
            'blockId': block_id,
            'convType': conv_type,
            'stride': layer_stride,
            'rate': layer_rate,
            'outputStride': current_stride # Looks like the variable 'outputStride' is never used anywhere.
        })
        block_id += 1

    return buff


def load_variables(chkpoint, base_dir=BASE_DIR):
    """
    Load all weights and biases from the C-struct binary files the manifest.json file refers to into tensorflow variables and
    attach those to the manifest data structure as property 'x' under their corresponding variable name.
    If no manifest is found, it will be downloaded first together with all the variable files it refers to.

    :param chkpoint: The checkpoint name. This name is important because it is part of the URL structure where the variables
    are downloaded from, and the name is reused on the local filesystem for consistency.
    :param base_dir: The local folder name where the posenet weights are downloaded in (usually in a temp folder).
    :return: The loaded content of the manifest is used as a data structure where the tensorflow variables created in this
    function are added to and hashed under the 'x' property of each variable.

    Note for refactoring: To make this function reusable for other networks, the weights downloader should be either
    1/ more generic, or 2/ extracted outside this function. Apart from this, this function is likely very reusable for other networks.
    """

    manifest_path = os.path.join(base_dir, chkpoint, "manifest.json")
    if not os.path.exists(manifest_path):
        print('Weights for checkpoint %s are not downloaded. Downloading to %s ...' % (chkpoint, base_dir))
        from posenet.converter.wget import download
        download(chkpoint, base_dir)
        assert os.path.exists(manifest_path)

    with open(manifest_path) as f:
        variables = json.load(f)

    # with tf.variable_scope(None, 'MobilenetV1'):
    for x in variables:
        filename = variables[x]["filename"]
        byte = open(os.path.join(base_dir, chkpoint, filename), 'rb').read()
        fmt = str(int(len(byte) / struct.calcsize('f'))) + 'f'
        d = struct.unpack(fmt, byte)
        d = tf.cast(d, tf.float32)
        d = tf.reshape(d, variables[x]["shape"])
        variables[x]["x"] = tf.Variable(d, name=x)

    return variables


def _read_imgfile(path, width, height):
    """
    Read an image file, resize it and normalize its values to match the MobileNetV1's expected input features.

    :param path: The path on the fs where the image is located.
    :param width: The requested image target width.
    :param height: The requested image target height.
    :return: The resized image with normalized pixels as a 3D array (height, width, channels).
    """

    img = cv2.imread(path)
    # The cv2.resize shape definition is indeed (width, height), while the image shape from cv2.imread is (height, width, channels).
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(float)
    img = img * (2.0 / 255.0) - 1.0
    return img


def build_network(image, layers, variables):
    """
    Build a tensorflow network instance based on the definition in the 'layers' parameter and the given variables.
    The layer names used are MobileNetV1 specific.

    Note: See how/if this can be made more generic to build other networks like MobileNetV2 / ResNet50 / ...

    :param image: The tensor placeholder that will be used to feed image data into the network. It's the starting point for the network.
    :param layers: The layer definitions as defined by the 'to_output_strided_layers' function.
    :param variables: The variables that instantiate the requested network. This parameter represents the network's manifest that
    was loaded from the manifest.json file and that was enriched with tensorflow variables that were loaded from the variable
    snapshot files the manifest refers to (by the 'load_variables' function).
    :return: The built tensorflow network.
    """

    def _weights(layer_name):
        return variables["MobilenetV1/" + layer_name + "/weights"]['x']

    def _biases(layer_name):
        return variables["MobilenetV1/" + layer_name + "/biases"]['x']

    def _depthwise_weights(layer_name):
        return variables["MobilenetV1/" + layer_name + "/depthwise_weights"]['x']

    def _conv_to_output(mobile_net_output, output_layer_name):
        w = tf.nn.conv2d(input=mobile_net_output, filters=_weights(output_layer_name), strides=[1, 1, 1, 1], padding='SAME')
        w = tf.nn.bias_add(w, _biases(output_layer_name), name=output_layer_name)
        return w

    def _conv(inputs, stride, block_id):
        return tf.nn.relu6(
            tf.nn.conv2d(input=inputs, filters=_weights("Conv2d_" + str(block_id)), strides=stride, padding='SAME')
            +
            _biases("Conv2d_" + str(block_id))
        )

    def _separable_conv(inputs, stride, block_id, dilations):
        if dilations is None:
            dilations = [1, 1]

        dw_layer = "Conv2d_" + str(block_id) + "_depthwise"
        pw_layer = "Conv2d_" + str(block_id) + "_pointwise"

        # 'NHWC' = data format [batch, height, width, channels]
        # The dilations are the number of repeated values in the height and width dimension to get a depthwise convolution.
        # A depthwise convolution uses a filter (kernel) with a depth of 1 instead of the channel depth to get fewer variables that
        # have to be learned, and so achieve a faster but less accurate network. When the rate (or dilation) is 1, then the strides
        # must all be 1, see: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/depthwise_conv2d
        w = tf.nn.depthwise_conv2d(input=inputs, filter=_depthwise_weights(dw_layer), strides=stride, padding='SAME', dilations=dilations, data_format='NHWC')
        w = tf.nn.bias_add(w, _biases(dw_layer))
        w = tf.nn.relu6(w)

        w = tf.nn.conv2d(input=w, filters=_weights(pw_layer), strides=[1, 1, 1, 1], padding='SAME')
        w = tf.nn.bias_add(w, _biases(pw_layer))
        w = tf.nn.relu6(w)

        return w

    x = image
    buff = [] # remove this buffer, seems like it's not used
    with tf.compat.v1.variable_scope(None, 'MobilenetV1'):

        for m in layers:
            stride = [1, m['stride'], m['stride'], 1]
            rate = [m['rate'], m['rate']]
            if m['convType'] == "conv2d":
                x = _conv(x, stride, m['blockId'])
                buff.append(x) # remove this buffer
            elif m['convType'] == "separableConv":
                x = _separable_conv(x, stride, m['blockId'], rate)
                buff.append(x) # remove this buffer

    heatmaps = _conv_to_output(x, 'heatmap_2')
    offsets = _conv_to_output(x, 'offset_2')
    displacement_fwd = _conv_to_output(x, 'displacement_fwd_2')
    displacement_bwd = _conv_to_output(x, 'displacement_bwd_2')
    heatmaps = tf.sigmoid(heatmaps, 'heatmap')
    # It looks like the outputs 'partheat', 'partoff' and 'segment' are not used.
    # It looks like only the '_2' variant is used of 'heatmap', 'offset', 'displacement_fwd' and 'displacement_bwd'.
    # To verify: Are the '_2' variants coupled to the choice of the outputstride of 16 in the config.yaml file?

    return heatmaps, offsets, displacement_fwd, displacement_bwd


def convert(model_id, model_dir, check=False):
    """
    Download and read the weight and bias variables for MobileNetV1, create the network and instantiate it with those variables.
    Then write the instantiated network to a model file and corresponding checkpoint files.

    :param model_id: Refers to the model to load, as defined in the config.yaml file.
    :param model_dir: Defines where the model and checkpoint files will be saved.
    :param check: Indicates whether or not to verify the model by feeding it a sample image.
    :return: Nothing, the model and checkpoint files are written to the filesystem.
    """

    cfg = load_config()
    checkpoints = cfg['checkpoints']
    image_size = cfg['imageSize']
    output_stride = cfg['outputStride'] # to verify: is this output_stride coupled to the downloaded weights? (current assumption is 'yes')
    chkpoint = checkpoints[model_id]

    if chkpoint == 'mobilenet_v1_050':
        mobile_net_arch = cfg['mobileNet50Architecture']
    elif chkpoint == 'mobilenet_v1_075':
        mobile_net_arch = cfg['mobileNet75Architecture']
    else:
        mobile_net_arch = cfg['mobileNet100Architecture']
    # The 'mobilenet_v1_101' seems to have the same architecture as 'mobileNet100Architecture'.

    width = image_size
    height = image_size

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    cg = tf.Graph()
    with cg.as_default():
        layers = to_output_strided_layers(mobile_net_arch, output_stride)
        variables = load_variables(chkpoint)

        init = tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session() as sess:
            sess.run(init)

            image_ph = tf.compat.v1.placeholder(tf.float32, shape=[1, None, None, 3], name='image')
            outputs = build_network(image_ph, layers, variables)

            sess.run(
                [outputs],
                feed_dict={
                    image_ph: [np.ndarray(shape=(height, width, 3), dtype=np.float32)]
                }
            )

            save_path = os.path.join(model_dir, 'model-%s' % chkpoint)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            builder = tf.compat.v1.saved_model.Builder(save_path)
            builder.add_meta_graph_and_variables(sess, tags=[tf.saved_model.SERVING])
            builder.save()

            if check and os.path.exists("./images/tennis_in_crowd.jpg"):
                # Result
                input_image = _read_imgfile("./images/tennis_in_crowd.jpg", width, height)
                input_image = np.array(input_image, dtype=np.float32)
                input_image = input_image.reshape(1, height, width, 3)

                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                    outputs,
                    feed_dict={image_ph: input_image}
                )

                print("Test image stats")
                print(input_image)
                print(input_image.shape)
                print(np.mean(input_image))

                heatmaps_result = heatmaps_result[0]

                print("Heatmaps")
                print(heatmaps_result[0:1, 0:1, :])
                print(heatmaps_result.shape)
                print(np.mean(heatmaps_result))
