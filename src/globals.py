from torchvision import transforms
from torch.autograd import Variable
from model import *
import torch
import config as cfg
import os
import json


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # Check cuda, if cuda gpu, if not cpu

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((cfg.IM_RESIZE, cfg.IM_RESIZE)),
    transforms.RandomCrop(cfg.IM_CROP),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((cfg.IM_CROP, cfg.IM_CROP)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().float()
    return Variable(x, volatile=volatile)


def to_var_labels(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().long()
    return Variable(x, volatile=volatile)


def load_dataset(set_name, model_name):
    json_path = cfg.MSASL_RGB_PATH + "/%s_%s_rgb.json" % (cfg.DATASET_NAME, set_name)
    set_path = cfg.MSASL_FLOW_PATH + "/%s" % (set_name)
    create_dir(cfg.TRAIN_MODEL_PATH)
    path_list, label_list = [], []

    # load json
    with open(json_path) as f:
        train_json = json.load(f)

    # traverse dataset list
    for vid in train_json:
        video_id = vid["videoId"]
        video_label = vid["label"]
        video_path = os.path.join(set_path, video_id)
        video_files = os.listdir(video_path)

        for i, video_file in enumerate(video_files):
            if model_name == 'spatial':
                if (i % 5) == 0:
                    path_list.append(os.path.join(video_path, video_file))
                    label_list.append(video_label)

            elif model_name == 'temporal':
                video_number = video_file.split('.')[0]
                if int(video_number) < len(video_files) - 10:
                    path_list.append(os.path.join(video_path, video_file))
                    label_list.append(video_label)

    return path_list, label_list


# TODO
def load_model(epoch):
    # Build models
    base_model = BaseModel(cfg.TEMPORAL_IN_CHANNEL, len(cfg.CLASSES)).eval()  # eval mode (batchnorm uses moving mean/variance)

    # use GPU if available.
    if torch.cuda.is_available():
        base_model.cuda()

    # Load the trained model parameters
    base_model.load_state_dict(torch.load(os.path.join(cfg.TRAIN_MODEL_PATH, 'spatial_model-%d.pkl' % epoch),
                                          map_location=lambda storage, loc: storage))

    return base_model


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)