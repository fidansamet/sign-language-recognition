from config import create_dir, MSASL_RGB_PATH, MSASL_FLOW_PATH
import cv2
import os
import numpy as np


def save_optical_flow():
    create_dir(MSASL_FLOW_PATH)

    for subset in ["train", "val", "test"]:
        root, directories, files = next(os.walk(MSASL_RGB_PATH + "/" + subset))
        create_dir(MSASL_FLOW_PATH + "/" + subset)

        for directory in directories:
            if directory == 'org':
                break
            create_dir(MSASL_FLOW_PATH + "/" + subset + "/" + directory + "/")
            minimal_flow(MSASL_RGB_PATH + "/" + subset + "/" + directory + "/",
                         MSASL_FLOW_PATH + "/" + subset + "/" + directory + "/")


def minimal_flow(rgb_dir, flow_dir):

    files = os.listdir(rgb_dir)
    if len(files) == 0:
        return

    prev_frame = cv2.imread(rgb_dir + "0.png")
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for file_name in range(1, len(files)):
        next_frame = cv2.imread(rgb_dir + "{}.png".format(file_name))
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

        if file_name == len(files) - 1:
            flow = calculate_flow(next_frame, next_frame)
            np.savez(os.path.join(flow_dir, "{}.npz".format(file_name - 1)), flow)
            break

        flow = calculate_flow(prev_frame, next_frame)
        np.savez(os.path.join(flow_dir + "{}.npz".format(file_name - 1)), flow)
        prev_frame = next_frame


def calculate_flow(prev_frame, next_frame):
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev_frame, next_frame, None)
    print("flow computed!")
    return flow
