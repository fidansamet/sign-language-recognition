from spatial_model.model import *
from spatial_model.dataloader import get_loader
import config as cfg
from torchvision import transforms
from torch.autograd import Variable
import json
import os
import datetime
import torch
import collections
import numpy as np


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().float()
    return Variable(x, volatile=volatile)

def to_var_labels(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda().long()
    return Variable(x, volatile=volatile)


def train():

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

    # print(train_json)

    # open loss info
    today = datetime.datetime.now()
    # loss_info = open(cfg.TRAIN_MODEL_PATH + '/loss_' + str(today) + '.txt', 'w')

    transform = transforms.Compose([
        transforms.Resize((cfg.IM_RESIZE, cfg.IM_RESIZE)),
        transforms.RandomCrop(cfg.IM_CROP),
        # transforms.RandomChoice(augmentations),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    # Build data loader
    data_loader = get_loader(image_path_list, image_label_list, cfg.BATCH_SIZE, shuffle=True, transform=transform, num_workers=cfg.NUM_WORKERS)

    # create model
    spatial_model = BaseModel(3, len(cfg.CLASSES))

    # use GPU if available.
    if torch.cuda.is_available():
        spatial_model.cuda()

    # daha once kaldiginiz bir noktadan baslamak icin.
    # if cfg.LOAD_TRAINED_MODEL:
    #     spatial_model.load_state_dict(torch.load(os.path.join(cfg.TRAIN_MODEL_PATH, cfg.LOAD_MODEL_NAME)))


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(spatial_model.parameters())),
                                 lr=cfg.LEARNING_RATE)

    loss_hist = collections.deque(maxlen=500)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(1, cfg.EPOCH_COUNT + 1):
        for i, (images, label) in enumerate(data_loader):
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            # Set mini-batch ground truth
            label = to_var_labels(label, volatile=False)
            # Forward, Backward and Optimize
            spatial_model.zero_grad()
            # feed images to CNN model
            predicted_label = spatial_model(images)

            loss = criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

            loss_hist.append(float(loss))

            # Print log info
            if i % cfg.LOG_STEP == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.7f, Running Loss: %5.6f'
                      % (epoch, cfg.EPOCH_COUNT, i, total_step,
                         loss.data, np.mean(loss_hist)))  #

                # loss_info.write('Epoch [%d/%d], Step [%d/%d], Loss: %.7f, Running Loss: %5.6f \n'
                #                 % (epoch, cfg.EPOCH_COUNT, i, total_step,
                #                    loss.data, np.mean(loss_hist)))

        # # Save the models
        # if epoch % cfg.SAVE_PERIOD_IN_EPOCHS == 0:
        #     torch.save(dream_model.state_dict(),
        #                os.path.join(cfg.TRAIN_MODEL_PATH,
        #                             'dream_model-%d.pkl' % epoch))


train()
