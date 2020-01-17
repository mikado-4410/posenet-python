import tensorflow as tf
import cv2
import time
import argparse

from posenet.posenet_factory import load_model
from posenet.utils import draw_skel_and_kp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():

    print('Tensorflow version: %s' % tf.__version__)
    assert tf.__version__.startswith('2.'), "Tensorflow version 2.x must be used!"

    model = 'posenet'  # posenet bodypix
    neuralnet = 'resnet50_v1'  # mobilenet_v1_100 resnet50_v1
    model_variant = 'stride32'  # stride16 stride32

    posenet = load_model(model, neuralnet, model_variant)

    if args.file is not None:
        cap = cv2.VideoCapture(args.file)
    else:
        cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0

    while True:
        res, img = cap.read()
        if not res:
            raise IOError("webcam failure")

        pose_scores, keypoint_scores, keypoint_coords = posenet.estimate_multiple_poses(img)

        overlay_image = draw_skel_and_kp(
            img, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()