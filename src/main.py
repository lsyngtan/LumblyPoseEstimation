"""Pose trainer main script."""

import argparse
import os, glob
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from parse import parse_sequence, load_ps
from evaluate import evaluate_pose

# ** Important setup notes: **
# 
# - Pose trainer is designed to run on Windows only, since it uses the Windows portable version of OpenPose.
# - Pose trainer expects the openpose folder to be in the pose trainer repository folder.
# - Pose trainer should be run from the root folder of the pose trainer repository.
#
# Please follow the instructions at 
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#windows-portable-demo
# to download and set up OpenPose, and move it into the root folder of the repository.
# (ex. If this repository is Documents/pose-trainer, then openpose will be Documents/pose-trainer/openpose)
# 
# OpenPose can run on a CPU-only machine, but it will be very slow.
#
# If you have a computer with an NVIDIA GPU, OpenPose will run significantly faster.

def main():
    parser = argparse.ArgumentParser(description='Pose Trainer')
    parser.add_argument('--mode', type=str, default='evaluate', help='Pose Trainer application mode.\n'
            'One of evaluate, batch_json, evaluate_npy. See the code for more info.')
    # parser.add_argument('--input_folder', type=str, default='videos', help='(Used by the batch_json mode only)\n'
    #         'Input folder for videos.\n'
    #         'Defaults to the videos folder in this repository folder.')
    # parser.add_argument('--output_folder', type=str, default='poses', help='(Used by the batch_json mode only)\n'
    #         'Folder for pose JSON files.\n'
    #         'Defaults to the poses folder in this repository folder.')
    # parser.add_argument('--video', type=str, help='(Used by the evaluate mode only)\n'
    #         'Input video filepath for evaluation. Looks for it in the root folder of the repository.')
    # parser.add_argument('--file', type=str, help='(Used by the evaluate_npy mode only)\n'
    #         'Full path to the input .npy file for evaluation.')
    parser.add_argument('--exercise', type=str, default='bird_dog', help='Exercise type to evaluate.')

    args = parser.parse_args()

    if args.mode == 'evaluate':
        # if args.video:
            # print('processing video file...')
            # video = os.path.basename(args.video)
            
            # # Run OpenPose on the video, and write a folder of JSON pose keypoints to a folder in
            # # the repository root folder with the same name as the input video.
            # output_path = os.path.join('..', os.path.splitext(video)[0])
            # openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
            # os.chdir('openpose')
            # subprocess.call([openpose_path, 
            #                 # Use the COCO model since it outputs the keypoint format pose trainer is expecting.
            #                 '--model_pose', 'COCO', 
            #                 # Use lower resolution for CPU only machines.
            #                 # If you're running the GPU version of OpenPose, or want to wait longer
            #                 # for higher quality, you can remove this line.
            #                 '--net_resolution', '-1x176', 
            #                 '--video', os.path.join('..', args.video), 
            #                 '--write_json', output_path])
            video_folders = glob.glob('C:/Users/jquin/Desktop/BME_Capstone_Lumbly/JSON/bird_dog_jsons/*/')
            for folder in video_folders:
                # Parse the pose JSON in the output path, and write it as a .npy file to the repository root folder.
                parse_sequence(folder, 'C:/Users/jquin/Desktop/BME_Capstone_Lumbly/bird_dog_npy')
                # Load the .npy pose sequence and evaluate the pose as the specified exercise.
                vid_dir = folder.split("\\")[-2]
                pose_seq = load_ps(f"C:/Users/jquin/Desktop/BME_Capstone_Lumbly/bird_dog_npy/{vid_dir}.npy")
                pkl_path = os.path.join("C:/Users/jquin/Desktop/BME_Capstone_Lumbly/bird_dog_pkls", vid_dir)
                (correct, feedback) = evaluate_pose(pose_seq, args.exercise, pkl_path)
                # if correct:
                #     print('Exercise performed correctly!')
                # else:
                #     print('Exercise could be improved:')
                # print(feedback)
        # else:
        #     print('No video file specified.')
        #     return
    else:
        print('Unrecognized mode option.')
        return




if __name__ == "__main__":
    main()
