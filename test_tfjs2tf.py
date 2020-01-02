import posenet.converter.tfjs2tf as tfjs2tf
import posenet.converter.tfjsdownload as tfjsdownload


def main():
    tfjsdownload.download_tfjs_model('bodypix', 'resnet50_v1', 'stride16')
    tfjs2tf.convert('bodypix', 'resnet50_v1', 'stride16')
    tfjsdownload.fix_model_file(tfjsdownload.model_config('bodypix', 'resnet50_v1', 'stride16'))
    tfjs2tf.list_tensors('posenet', 'resnet50_v1', 'stride16')
    tfjs2tf.list_tensors('bodypix', 'mobilenet_v1_100', 'stride16')
    tfjs2tf.list_tensors('posenet', 'mobilenet_v1_100', 'stride16')

# have a look at: https://github.com/tensorflow/tfjs/tree/master/tfjs-converter
# https://github.com/patlevin/tfjs-to-tf

## BodyPix

# see: https://stackoverflow.com/questions/58841355/bodypix-real-time-person-segmentation/59509874#59509874
#
# https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/model-stride16.json
# see weightsManifest.paths for the shard names:
# https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/group1-shard1of23.bin
# ...
# https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/resnet50/float/group1-shard23of23.bin


# https://storage.googleapis.com/tfjs-models/savedmodel/bodypix/mobilenet/float/100/model-stride16.json
# see weightsManifest.paths
# "group1-shard1of4.bin",
# "group1-shard2of4.bin",
# "group1-shard3of4.bin",
# "group1-shard4of4.bin"


## PoseNet

# https://storage.googleapis.com/tfjs-models/savedmodel/posenet/mobilenet/float/100/model-stride16.json
# https://storage.googleapis.com/tfjs-models/savedmodel/posenet/resnet50/float/model-stride16.json


# Old model format
# https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_101/manifest.json

