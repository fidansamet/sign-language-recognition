import json
import os
import youtube_dl
from constants import DATASET_NAME, TRAIN_JSON_NAME, VAL_JSON_NAME, TEST_JSON_NAME, CLASSES
import subprocess
import cv2


def download_dataset():
    create_dir(DATASET_NAME)
    save_videos("train", TRAIN_JSON_NAME)
    save_videos("val", VAL_JSON_NAME)
    save_videos("test", TEST_JSON_NAME)


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def save_frames(file_name, dir_name, file_count = 0):
    video = cv2.VideoCapture(file_name)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(dir_name + '/' + str(file_count) + '.jpg', frame)
        file_count += 1

    video.release()
    cv2.destroyAllWindows()


def save_videos(folder_name, json_name, video_id= 0, json_data=[]):
    create_dir(DATASET_NAME + "/" + folder_name)        # create train, val or test directory
    file = open(json_name, )
    data = json.load(file)

    for i in data:
        cur_str = json.dumps(i)
        cur_row = json.loads(cur_str)

        if cur_row["clean_text"] in CLASSES:

            dir_name = DATASET_NAME + "/" + folder_name + "/" + str(video_id)
            file_name = DATASET_NAME + "/" + folder_name + '/%s_%d.mp4' % (DATASET_NAME, video_id)

            create_dir(dir_name)       # create video subdir

            print(cur_row["url"])
            start_time = cur_row["start_time"]
            end_time = cur_row["end_time"]

            ydl_opts = {
                'outtmpl': DATASET_NAME + "/" + folder_name + '/org/%s_%d' % (DATASET_NAME, video_id) + ".%(ext)s",
                'format': 'bestvideo[ext=mp4]', "merge_output_format": 'mp4', 'ignoreerrors': True}

            # download video
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url_list=[cur_row["url"]])

            # crop video
            subprocess.call(['ffmpeg', '-y', '-i',
                             DATASET_NAME + "/" + folder_name + '/org/%s_%d' % (DATASET_NAME, video_id) + ".mp4",
                             '-ss', str(start_time), '-t', str(end_time - start_time), file_name])

            # open video
            save_frames(file_name, dir_name)

            # save json
            json_data.append({
                "videoId": '%d' % video_id,
                "cleanText": cur_row["clean_text"],
                "label": cur_row["label"],
                "width": cur_row["width"],
                "height": cur_row["height"]
            })

            video_id += 1

    file.close()
    with open(DATASET_NAME + "/%s_%s_info.json" % (DATASET_NAME, folder_name), "w") as outfile:
        json.dump(json_data, outfile)
