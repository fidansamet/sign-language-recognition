import json
import youtube_dl
import config as cfg
import subprocess
import cv2


def download_dataset():
    cfg.create_dir(cfg.MSASL_RGB_PATH)
    save_videos("train", cfg.TRAIN_JSON_PATH)
    save_videos("val", cfg.VAL_JSON_PATH)
    save_videos("test", cfg.TEST_JSON_PATH)


def save_frames(file_name, dir_name, file_count=0):
    video = cv2.VideoCapture(file_name)

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imwrite(dir_name + '/' + str(file_count) + '.png', frame)
        file_count += 1

    video.release()
    cv2.destroyAllWindows()


def save_videos(folder_name, json_name, video_id= 0, json_data=[]):
    cfg.create_dir(cfg.MSASL_RGB_PATH + "/" + folder_name)        # create train, val or test directory
    file = open(json_name, )
    data = json.load(file)

    for i in data:
        cur_str = json.dumps(i)
        cur_row = json.loads(cur_str)

        if cur_row["clean_text"] in cfg.CLASSES:

            dir_name = cfg.MSASL_RGB_PATH + "/" + folder_name + "/" + str(video_id)
            file_name = cfg.MSASL_RGB_PATH + "/" + folder_name + '/%s_%d.mp4' % (cfg.DATASET_NAME, video_id)

            cfg.create_dir(dir_name)       # create video subdir

            print(cur_row["url"])
            start_time = cur_row["start_time"]
            end_time = cur_row["end_time"]

            ydl_opts = {
                'outtmpl': cfg.MSASL_RGB_PATH + "/" + folder_name + '/org/%s_%d' % (cfg.DATASET_NAME, video_id) + ".%(ext)s",
                'format': 'bestvideo[ext=mp4]', "merge_output_format": 'mp4', 'ignoreerrors': True}

            # download video
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url_list=[cur_row["url"]])

            # crop video
            subprocess.call(['ffmpeg', '-y', '-i',
                             cfg.MSASL_RGB_PATH + "/" + folder_name + '/org/%s_%d' % (cfg.DATASET_NAME, video_id) + ".mp4",
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
    with open(cfg.MSASL_RGB_PATH + "/%s_%s_rgb.json" % (cfg.DATASET_NAME, folder_name), "w") as outfile:
        json.dump(json_data, outfile)
