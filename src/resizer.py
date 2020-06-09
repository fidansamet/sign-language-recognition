import config as cfg
import cv2
import os
import json
from globals import create_dir


def resize_frame(frame):
    h, w, _ = frame.shape
    if h < w:
        resize_ratio = cfg.MIN_RESIZE / h
        h = cfg.MIN_RESIZE
        w = w * resize_ratio
    else:
        resize_ratio = cfg.MIN_RESIZE / w
        w = cfg.MIN_RESIZE
        h = h * resize_ratio

    resized_image = cv2.resize(frame, (int(w), int(h)))
    return resized_image


create_dir(cfg.RESIZED_RGB_PATH)


create_dir(cfg.RESIZED_RGB_PATH + "/test")
SAVE_PATH = cfg.RESIZED_RGB_PATH + "/test"
train_json_path = cfg.MSASL_RGB_PATH + "/%s_test_rgb.json" % (cfg.DATASET_NAME)
train_path = cfg.MSASL_RGB_PATH + "/test"

# load json
with open(train_json_path) as f:
    train_json = json.load(f)


# traverse dataset list
for vid in train_json:
    video_id = vid["videoId"]
    video_path = os.path.join(train_path, video_id)
    video_files = os.listdir(video_path)
    create_dir(SAVE_PATH + "/" + video_id)

    for video_file in video_files:
        img = cv2.imread(os.path.join(video_path, video_file))
        img = resize_frame(img)
        cv2.imwrite(os.path.join(os.path.join(SAVE_PATH, video_id), video_file), img)
