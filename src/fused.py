from model import *
from globals import TRAIN_TRANSFORM, VAL_TRANSFORM, load_dataset, to_var, to_var_labels, load_model
from dataloader import get_spatial_loader, get_temporal_loader
import torch.nn.functional as F
import config as cfg
import os
import datetime
import torch
import collections
import numpy as np
from sklearn import metrics


def get_data_loaders(model_name, train_path_list, train_label_list, val_path_list, val_label_list):
    # build data loader
    s_train_data_loader = get_spatial_loader(train_path_list, train_label_list, cfg.BATCH_SIZE, shuffle=True,
                                             transform=TRAIN_TRANSFORM, num_workers=cfg.NUM_WORKERS)
    s_val_data_loader = get_spatial_loader(val_path_list, val_label_list, 1, shuffle=False,
                                           transform=VAL_TRANSFORM, num_workers=1)

    t_train_data_loader = get_temporal_loader(train_path_list, train_label_list, cfg.BATCH_SIZE, shuffle=True,
                                              transform=TRAIN_TRANSFORM, num_workers=cfg.NUM_WORKERS)
    t_val_data_loader = get_temporal_loader(val_path_list, val_label_list, 1, shuffle=False,
                                            transform=VAL_TRANSFORM, num_workers=1)
    model = FusedModel()

    return model, s_train_data_loader, s_val_data_loader, t_train_data_loader, t_val_data_loader


def train(model_name):
    # load dataset
    train_path_list, train_label_list = load_dataset('train', model_name)
    val_path_list, val_label_list = load_dataset('val', model_name)

    # open loss files
    today = datetime.datetime.now()
    s_train_loss_info = open(cfg.TRAIN_MODEL_PATH + '/s_train_loss_' + str(today) + '.txt', 'w')
    t_train_loss_info = open(cfg.TRAIN_MODEL_PATH + '/t_train_loss_' + str(today) + '.txt', 'w')
    val_loss_info = open(cfg.TRAIN_MODEL_PATH + '/val_loss_' + str(today) + '.txt', 'w')

    # get dataloaders
    model, s_train_data_loader, s_val_data_loader, t_train_data_loader, t_val_data_loader = get_data_loaders(model_name,
                                                                                                             train_path_list,
                                                                                                             train_label_list,
                                                                                                             val_path_list,
                                                                                                             val_label_list)

    # use GPU if available.
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    s_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.spatial_model.parameters())), lr=cfg.LEARNING_RATE)
    t_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.temporal_model.parameters())), lr=cfg.LEARNING_RATE)

    loss_hist = collections.deque(maxlen=500)

    # train the model
    total_step = len(s_train_data_loader)

    for epoch in range(1, cfg.EPOCH_COUNT + 1):
        for i, (s_data, t_data) in enumerate(zip(s_train_data_loader, t_train_data_loader)):

            s_imgs, s_labels = s_data
            t_arrs, t_labels = t_data

            # Set mini-batch dataset and ground truth
            s_imgs, s_labels = to_var(s_imgs, volatile=True), to_var_labels(s_labels, volatile=False)
            t_arrs, t_labels = to_var(t_arrs, volatile=True), to_var_labels(t_labels, volatile=False)

            # zero the parameter gradients
            # s_optimizer.zero_grad()
            # t_optimizer.zero_grad()

            # Forward, Backward and Optimize
            model.spatial_model.zero_grad()
            model.temporal_model.zero_grad()

            # feed images to CNN model
            s_predicted_labels = model.spatial_model(s_imgs)
            t_predicted_labels = model.temporal_model(t_arrs)

            s_loss = criterion(s_predicted_labels, s_labels)
            t_loss = criterion(t_predicted_labels, t_labels)

            combined_loss = s_loss + t_loss
            combined_loss.backward()
            s_optimizer.step()
            t_optimizer.step()

            loss_hist.append(float(combined_loss))

            # Print log info
            if i % cfg.LOG_STEP == 0:
                print('Epoch [%d/%d], Step [%d/%d], Spatial Loss: %.7f, Temporal Loss: %.7f, Running Loss: %5.6f'
                      % (epoch, cfg.EPOCH_COUNT, i, total_step,
                         s_loss.data, t_loss.data, np.mean(loss_hist)))

                s_train_loss_info.write('Epoch [%d/%d], Step [%d/%d], Spatial Loss: %.7f, Running Loss: %5.6f \n'
                                      % (epoch, cfg.EPOCH_COUNT, i, total_step,
                                         s_loss.data, np.mean(loss_hist)))

                t_train_loss_info.write('Epoch [%d/%d], Step [%d/%d], Temporal Loss: %.7f, Running Loss: %5.6f \n'
                                      % (epoch, cfg.EPOCH_COUNT, i, total_step,
                                         t_loss.data, np.mean(loss_hist)))

        # Save the models
        if epoch % cfg.SAVE_PERIOD_IN_EPOCHS == 0:
            # run test on val set
            test_acc, test_loss = test(model, s_val_data_loader, t_val_data_loader, criterion)
            val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f, Val Accuracy: %.3f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step, test_loss, test_acc))

            torch.save(model.state_dict(), os.path.join(cfg.TRAIN_MODEL_PATH, 'spatial_model-%d.pkl' % epoch))


def test(model, s_loader, t_loader, criterion):

    test_loss = 0,
    correct = 0
    s_ground_truths, s_test_results = [], []
    t_ground_truths, t_test_results = [], []
    model.eval()

    with torch.no_grad():
        for i, (s_data, t_data) in enumerate(zip(s_loader, t_loader)):

            s_imgs, s_labels = s_data
            t_arrs, t_labels = t_data

            # Set mini-batch dataset and ground truth
            s_imgs, s_labels = to_var(s_imgs, volatile=True), to_var_labels(s_labels, volatile=False)
            t_arrs, t_labels = to_var(t_arrs, volatile=True), to_var_labels(t_labels, volatile=False)

            for j in s_labels.tolist():
                s_ground_truths.append(j)

            for j in t_labels.tolist():
                t_ground_truths.append(j)

            s_predicted_labels = model.spatial_model(s_imgs)
            t_predicted_labels = model.temporal_model(t_arrs)

            s_outputs = F.softmax(s_predicted_labels)
            t_outputs = F.softmax(t_predicted_labels)

            outputs = (s_outputs + t_outputs) / 2.0

            # s_loss = criterion(s_outputs, s_labels)
            # t_loss = criterion(t_outputs, t_labels)

            # combined_loss = s_loss + t_loss
            # test_loss += combined_loss.cpu().data.numpy()   # sum up batch loss

            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # t_pred = t_outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(labels.view_as(pred)).sum().item()       # TODO

            for j in s_predicted_labels.tolist():
                s_test_results.append(j[0])

            for j in t_predicted_labels.tolist():
                t_test_results.append(j[0])


    # test_loss /= len(s_loader)
    test_acc = 100. * correct / len(s_loader)
    print(test_acc)

    score = metrics.accuracy_score(s_ground_truths, s_test_results)
    cls_report = metrics.classification_report(s_ground_truths, s_test_results)
    conf_mat = metrics.confusion_matrix(s_ground_truths, s_test_results)
    print("Accuracy of Spatial = " + str(score))
    print(cls_report)
    print(conf_mat)

    score = metrics.accuracy_score(t_ground_truths, t_test_results)
    cls_report = metrics.classification_report(t_ground_truths, t_test_results)
    conf_mat = metrics.confusion_matrix(t_ground_truths, t_test_results)
    print("Accuracy of Temporal = " + str(score))
    print(cls_report)
    print(conf_mat)

    return test_acc


def run_test(model_name):
    set_name = 'test'
    test_path_list, test_label_list = load_dataset(set_name, model_name)

    # build data loader
    s_test_data_loader = get_spatial_loader(test_path_list, test_label_list, 1, shuffle=False,
                                             transform=VAL_TRANSFORM, num_workers=1)

    t_test_data_loader = get_temporal_loader(test_path_list, test_label_list, 1, shuffle=False,
                                              transform=VAL_TRANSFORM, num_workers=1)

    model = load_model(90)
    criterion = nn.CrossEntropyLoss()
    test(model, s_test_data_loader, t_test_data_loader, criterion)
