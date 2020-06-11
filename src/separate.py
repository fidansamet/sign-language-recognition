from model import *
from globals import TRAIN_TRANSFORM, VAL_TRANSFORM, load_dataset, to_var, to_var_labels
from dataloader import get_spatial_loader, get_temporal_loader
import config as cfg
import torch.nn.functional as F
import os
import datetime
import torch
import collections
import numpy as np
from sklearn import metrics


def get_data_loaders(model_name, train_path_list, train_label_list, val_path_list, val_label_list):
    # build data loader
    if model_name == 'spatial':
        data_loader_train = get_spatial_loader(train_path_list, train_label_list, cfg.BATCH_SIZE, shuffle=True,
                                               transform=TRAIN_TRANSFORM, num_workers=cfg.NUM_WORKERS)
        data_loader_val = get_spatial_loader(val_path_list, val_label_list, 1, shuffle=False,
                                             transform=VAL_TRANSFORM, num_workers=1)
        model = BaseModel(cfg.SPATIAL_IN_CHANNEL, len(cfg.CLASSES), cfg.SPATIAL_FLATTEN)

    else:   # temporal
        data_loader_train = get_temporal_loader(train_path_list, train_label_list, cfg.BATCH_SIZE, shuffle=True,
                                                transform=TRAIN_TRANSFORM, num_workers=cfg.NUM_WORKERS)
        data_loader_val = get_temporal_loader(val_path_list, val_label_list, 1, shuffle=False,
                                              transform=VAL_TRANSFORM, num_workers=1)
        model = BaseModel(cfg.TEMPORAL_IN_CHANNEL, len(cfg.CLASSES), cfg.TEMPORAL_FLATTEN)

    return model, data_loader_train, data_loader_val


def train(model_name):
    # load dataset
    train_path_list, train_label_list = load_dataset('train', model_name)
    val_path_list, val_label_list = load_dataset('val', model_name)

    # open loss files
    today = datetime.datetime.now()
    train_loss_info = open(cfg.TRAIN_MODEL_PATH + '/loss_train_' + str(today) + '.txt', 'w')
    val_loss_info = open(cfg.TRAIN_MODEL_PATH + '/loss_val_' + str(today) + '.txt', 'w')

    # get dataloaders
    model, data_loader_train, data_loader_val = get_data_loaders(model_name, train_path_list, train_label_list, val_path_list, val_label_list)

    # use GPU if available.
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.parameters())), lr=cfg.LEARNING_RATE)

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
            model.zero_grad()
            # feed images to CNN model
            predicted_labels = model(images)

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
            test_acc, test_loss = test(model, data_loader_val, criterion)
            val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f, Val Accuracy: %.3f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step,
                                   test_loss, test_acc))

            torch.save(model.state_dict(),
                       os.path.join(cfg.TRAIN_MODEL_PATH,
                                    'spatial_model-%d.pkl' % epoch))


def test(model, validation_loader, criterion):

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
            outputs = F.softmax(outputs)
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

    return test_acc, test_loss


def run_test(model_name):
    set_name = 'test'
    test_path_list, test_label_list = load_dataset(set_name, model_name)

    # build data loader
    if model_name == 'spatial':
        data_loader_test = get_spatial_loader(test_path_list, test_label_list, 1, shuffle=False,
                                              transform=VAL_TRANSFORM, num_workers=1)

    else:    # temporal
        data_loader_test = get_temporal_loader(test_path_list, test_label_list, 1, shuffle=False,
                                               transform=VAL_TRANSFORM, num_workers=1)

    model = load_model(model_name, 90)
    criterion = nn.CrossEntropyLoss()
    test(model, data_loader_test, criterion)
