import sys
sys.path.insert(0,'..')
from model import *
from dataloader import get_spatial_loader
import config as cfg
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import json
import os
import datetime
import torch
import collections
import numpy as np
from sklearn import metrics


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


def load_dataset(set_name):
    json_path = cfg.MSASL_RGB_PATH + "/%s_%s_rgb.json" % (cfg.DATASET_NAME, set_name)
    set_path = cfg.MSASL_RGB_PATH + "/%s" % (set_name)
    cfg.create_dir(cfg.TRAIN_MODEL_PATH)
    img_path_list, img_label_list = [], []

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
            if (i % 5) == 0:
                img_path_list.append(os.path.join(video_path, video_file))
                img_label_list.append(video_label)

    return img_path_list, img_label_list


def train():
    # load dataset
    train_img_path_list, train_img_label_list = load_dataset('train')
    val_img_path_list, val_img_label_list = load_dataset('val')

    # open loss files
    today = datetime.datetime.now()
    train_loss_info = open(cfg.TRAIN_MODEL_PATH + '/loss_train_' + str(today) + '.txt', 'w')
    val_loss_info = open(cfg.TRAIN_MODEL_PATH + '/loss_val_' + str(today) + '.txt', 'w')

    # build data loader
    data_loader_train = get_spatial_loader(train_img_path_list, train_img_label_list, cfg.BATCH_SIZE, shuffle=True, transform=TRAIN_TRANSFORM, num_workers=cfg.NUM_WORKERS)
    data_loader_val = get_spatial_loader(val_img_path_list, val_img_label_list, 1, shuffle=False, transform=VAL_TRANSFORM, num_workers=1)

    # create model
    spatial_model = BaseModel(3, len(cfg.CLASSES))

    # use GPU if available.
    if torch.cuda.is_available():
        spatial_model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(spatial_model.parameters())), lr=cfg.LEARNING_RATE)

    loss_hist = collections.deque(maxlen=500)

    # train the model
    total_step = len(data_loader_train)
    for epoch in range(1, cfg.EPOCH_COUNT + 1):
        for i, (images, labels) in enumerate(data_loader_train):
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            # Set mini-batch ground truth
            labels = to_var_labels(labels, volatile=False)
            # Forward, Backward and Optimize
            spatial_model.zero_grad()
            # feed images to CNN model
            predicted_labels = spatial_model(images)

            loss = criterion(predicted_labels, labels)
            loss.backward()
            optimizer.step()

            loss_hist.append(float(loss))

            # Print log info
            if i % cfg.LOG_STEP == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.7f, Running Loss: %5.6f'
                      % (epoch, cfg.EPOCH_COUNT, i, total_step,
                         loss.data, np.mean(loss_hist)))  #

                train_loss_info.write('Epoch [%d/%d], Step [%d/%d], Loss: %.7f, Running Loss: %5.6f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step,
                                   loss.data, np.mean(loss_hist)))

        # Save the models
        if epoch % cfg.SAVE_PERIOD_IN_EPOCHS == 0:
            # run test on val set
            test_acc, test_loss, ground_truths, test_results = test(spatial_model, data_loader_val, criterion, set_name='val')
            val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f, Val Accuracy: %.3f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step,
                                   test_loss, test_acc))

            torch.save(spatial_model.state_dict(),
                       os.path.join(cfg.TRAIN_MODEL_PATH,
                                    'spatial_model-%d.pkl' % epoch))


def test(model, validation_loader, criterion, set_name='val'):

    test_loss = 0,
    correct = 0
    ground_truths = []
    test_results = []
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(validation_loader):
            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            # Set mini-batch ground truth
            labels = to_var_labels(labels, volatile=False)

            for ii in labels.tolist():
                ground_truths.append(ii)

            outputs = model(images)
            softmaxx = F.softmax(outputs)
            #test_loss += F.nll_loss(outputs, labels).item()  # sum up batch loss
            # test_loss += F.binary_cross_entropy(outputs, labels, reduction='sum').item()  # sum up batch loss
            loss = criterion(outputs, labels)
            #print(loss.cpu().data.numpy())
            test_loss += loss.cpu().data.numpy()   # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            # correct += pred.eq(labels.argmax(dim=1, keepdim=True)).sum().item()

            for iii in pred.tolist():
                test_results.append(iii[0])

    test_loss /= len(validation_loader)
    test_acc = 100. * correct / len(validation_loader)
    print(test_acc)

    score = metrics.accuracy_score(ground_truths, test_results)
    cls_report = metrics.classification_report(ground_truths, test_results)
    conf_mat = metrics.confusion_matrix(ground_truths, test_results)

    print("Accuracy = " + str(score))
    print(cls_report)
    print(conf_mat)

    # print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     set_name, test_loss, correct, len(validation_loader), test_acc))

    return test_acc, test_loss, ground_truths, test_results


def load_model(epoch):
    # Build models
    base_model = BaseModel(3, len(cfg.CLASSES)).eval()  # eval mode (batchnorm uses moving mean/variance)

    # use GPU if available.
    if torch.cuda.is_available():
        base_model.cuda()

    # Load the trained model parameters
    base_model.load_state_dict(torch.load(os.path.join(cfg.TRAIN_MODEL_PATH, 'spatial_model-%d.pkl' % epoch), map_location=lambda storage, loc: storage))
    
    return base_model


def run_test():

    set_name = 'test'
    image_path_list_test, image_label_list_test = load_dataset(set_name)
    # build val data loader
    data_loader_test = get_spatial_loader(image_path_list_test, image_label_list_test, 1, shuffle=False, transform=VAL_TRANSFORM, num_workers=1)

    base_model = load_model(10)
    criterion = nn.CrossEntropyLoss()

    test(base_model, data_loader_test, criterion, set_name=set_name)


#train()
run_test()
