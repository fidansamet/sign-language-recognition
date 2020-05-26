import json
import os
import youtube_dl
from constants import DATASET_NAME, TRAIN_JSON_NAME, VAL_JSON_NAME, TEST_JSON_NAME, CLASSES
import subprocess


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


def save_videos(folder_name, json_name, file_count=0, json_data=[]):
    create_dir(DATASET_NAME + "/" + folder_name)        # create train, val or test directory
    file = open(json_name, )
    data = json.load(file)

    for i in data:
        cur_str = json.dumps(i)
        cur_row = json.loads(cur_str)

        if cur_row["clean_text"] in CLASSES:
            start_time = cur_row["start_time"]
            end_time = cur_row["end_time"]
            # height = cur_row["height"]
            # width = cur_row["width"]
            # box = cur_row["box"]
            # x0, y0 = box[1] * width, box[0] * height
            # x1, y1 = box[3] * width, box[2] * height

            file_name = DATASET_NAME + "/" + folder_name + '/%s_%d.mp4' % (DATASET_NAME, file_count)
            ydl_opts = {'outtmpl': DATASET_NAME + "/" + folder_name + '/org/%s_%d' % (DATASET_NAME, file_count) + ".%(ext)s",
                        'format': 'bestvideo[ext=mp4]', "merge_output_format": 'mp4', 'ignoreerrors': True}

            print(cur_row["url"])
            # download video
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url_list=[cur_row["url"]])

            # crop video
            subprocess.call(['ffmpeg', '-y', '-i',
                             DATASET_NAME + "/" + folder_name + '/org/%s_%d' % (DATASET_NAME, file_count) + ".mp4",
                             '-ss', str(start_time), '-t', str(end_time - start_time), file_name])
            #  '-filter:v', 'crop=%d:%d:%d:%d'%((x1-x0), (y1-y0), x0, y0)

            json_data.append({
                "fileName": '%s_%d.mp4' % (DATASET_NAME, file_count),
                "cleanText": cur_row["clean_text"],
                "label": cur_row["label"],
                "startTime": start_time,
                "endTime": end_time,
                "startFrame": cur_row["start"],
                "endFrame": cur_row["end"],
                "fps": cur_row["fps"],
                "width": cur_row["width"],
                "height": cur_row["height"]
            })

            file_count += 1

    file.close()
    with open(DATASET_NAME + "/%s_%s_info.json" % (DATASET_NAME, folder_name), "w") as outfile:
        json.dump(json_data, outfile)
