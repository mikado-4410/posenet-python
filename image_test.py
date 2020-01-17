import tensorflow as tf
import cv2
import time
import argparse
import os

import posenet
from posenet.posenet_factory import load_model
import posenet.converter.tfjsdownload as tfjsdownload

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()


def main():

    print('Tensorflow version: %s' % tf.__version__)
    assert tf.__version__.startswith('2.'), "Tensorflow version 2.x must be used!"

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    model = 'posenet'  # posenet bodypix
    neuralnet = 'mobilenet_v1_100'  # mobilenet_v1_100 resnet50_v1
    model_variant = 'stride16'  # stride16 stride32

    posenet = load_model(model, neuralnet, model_variant)

    filenames = [f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    for f in filenames:
        img = cv2.imread(f)
        pose_scores, keypoint_scores, keypoint_coords = posenet.estimate_multiple_poses(img)
        img_poses = posenet.draw_poses(img, pose_scores, keypoint_scores, keypoint_coords)
        posenet.print_scores(f, pose_scores, keypoint_scores, keypoint_coords)
        cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), img_poses)

    print('Average FPS:', len(filenames) / (time.time() - start))

    posenet.close()


if __name__ == "__main__":
    main()
