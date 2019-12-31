import posenet.converter.tfjs2tf as converter


def main():
    converter.convert()

# have a look at: https://github.com/tensorflow/tfjs/tree/master/tfjs-converter

# see: https://stackoverflow.com/questions/58841355/bodypix-real-time-person-segmentation/59509874#59509874
# https://github.com/patlevin/tfjs-to-tf
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


# https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_101/manifest.json


