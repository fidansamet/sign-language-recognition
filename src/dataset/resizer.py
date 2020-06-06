train_json_path = cfg.MSASL_RGB_PATH + "/%s_train_rgb.json" % (cfg.DATASET_NAME)
    train_path = cfg.MSASL_RGB_PATH + "/train"

    # load json
    with open(train_json_path) as f:
        train_json = json.load(f)

    image_path_list, image_label_list = [], []

    # traverse dataset list
    for vid in train_json:
        video_id = vid["videoId"]
        video_label = vid["label"]
        video_path = os.path.join(train_path, video_id)
        video_files = os.listdir(video_path)

        for video_file in video_files:
            image_path_list.append(os.path.join(video_path, video_file))
            image_label_list.append(video_label)
