from model import *
from globals import TRAIN_TRANSFORM, VAL_TRANSFORM, load_fused_dataset, to_var, to_var_labels, load_fusion_model
from dataloader import get_fused_loader
import torch.nn.functional as F
import config as cfg
import os
import datetime
import torch
import collections
import numpy as np
from sklearn import metrics


def get_data_loaders(train_img_path_list, train_flow_path_list, train_label_list,
                     val_img_path_list, val_flow_path_list, val_label_list):
    # build data loader
    train_data_loader = get_fused_loader(train_img_path_list, train_flow_path_list, train_label_list,
                                         cfg.BATCH_SIZE, shuffle=True, transform=TRAIN_TRANSFORM,
                                         num_workers=cfg.NUM_WORKERS)
    val_data_loader = get_fused_loader(val_img_path_list, val_flow_path_list, val_label_list, 1, shuffle=False,
                                       transform=VAL_TRANSFORM, num_workers=1)

    return train_data_loader, val_data_loader


def train(model_name, pretrained):
    # load dataset
    train_img_path_list, train_flow_path_list, train_label_list = load_fused_dataset('train')
    val_img_path_list, val_flow_path_list, val_label_list = load_fused_dataset('val')

    # open loss files
    today = datetime.datetime.now()
    train_loss_info = open(cfg.TRAIN_MODEL_PATH + '/train_loss_' + str(today) + '.txt', 'w')
    val_loss_info = open(cfg.TRAIN_MODEL_PATH + '/val_loss_' + str(today) + '.txt', 'w')
    train_acc_loss_info = open(cfg.TRAIN_MODEL_PATH + '/train_acc_loss_' + str(today) + '.txt', 'w')

    # get dataloaders
    train_data_loader, val_data_loader = get_data_loaders(train_img_path_list, train_flow_path_list,
                                                         train_label_list, val_img_path_list,
                                                          val_flow_path_list, val_label_list)

    criterion = nn.CrossEntropyLoss()
    loss_hist = collections.deque(maxlen=500)

    if model_name == "late_fusion":
        late_fusion_train_epoch(train_data_loader, val_data_loader, criterion, loss_hist, train_loss_info,
                                val_loss_info, train_acc_loss_info, pretrained)
    else:
        early_fusion_train_epoch(train_data_loader, val_data_loader, criterion, loss_hist, train_loss_info,
                                 val_loss_info, train_acc_loss_info, pretrained)


def late_fusion_train_epoch(train_data_loader, val_data_loader, criterion, loss_hist, train_loss_info, val_loss_info, train_acc_loss_info, pretrained):
    model = FusedModel(cfg.LATE, pretrained=pretrained)

    # use GPU if available.
    if torch.cuda.is_available():
        model.cuda()

    s_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.spatial_model.parameters())),
                                   lr=cfg.LEARNING_RATE)
    t_optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.temporal_model.parameters())),
                                   lr=cfg.LEARNING_RATE)

    # train the model
    total_step = len(train_data_loader)

    test_acc, test_loss = late_fusion_test(model, val_data_loader, criterion)
    val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f,  Val Accuracy: %.3f \n'
                        % (0, cfg.EPOCH_COUNT, 0, total_step, test_loss, test_acc))

    test_acc, test_loss = late_fusion_test(model, train_data_loader, criterion)
    train_acc_loss_info.write('Epoch [%d/%d], Step [%d/%d], Train Loss: %.7f, Train Accuracy: %.3f \n'
                              % (0, cfg.EPOCH_COUNT, 0, total_step, test_loss, test_acc))

    for epoch in range(1, cfg.EPOCH_COUNT + 1):
        for i, (images, flows, labels) in enumerate(train_data_loader):
            # Set mini-batch images
            images = to_var(images, volatile=True)
            # Set mini-batch optical flows
            flows = to_var(flows, volatile=True)
            # Set mini-batch ground truth
            labels = to_var_labels(labels, volatile=False)

            # zero the parameter gradients
            # s_optimizer.zero_grad()
            # t_optimizer.zero_grad()

            # Forward, Backward and Optimize
            model.spatial_model.zero_grad()
            model.temporal_model.zero_grad()

            # feed images to CNN model
            s_predicted_labels = model.spatial_model(images)
            t_predicted_labels = model.temporal_model(flows)

            s_loss = criterion(s_predicted_labels, labels)
            t_loss = criterion(t_predicted_labels, labels)

            combined_loss = s_loss + t_loss
            combined_loss.backward()
            s_optimizer.step()
            t_optimizer.step()

            loss_hist.append(float(combined_loss))

            # Print log info
            if i % cfg.LOG_STEP == 0:
                print(
                    'Epoch [%d/%d], Step [%d/%d], Spatial Loss: %.7f, Temporal Loss: %.7f, Running Loss: %5.6f, Combined Loss: %.7f'
                    % (epoch, cfg.EPOCH_COUNT, i, total_step, s_loss.data, t_loss.data, np.mean(loss_hist),
                       combined_loss.data))

                train_loss_info.write('Epoch [%d/%d], Step [%d/%d], Spatial Loss: %.7f, Temporal Loss: %.7f,'
                                      'Running Loss: %5.6f, Combined Loss: %.7f \n'
                                      % (epoch, cfg.EPOCH_COUNT, i, total_step, s_loss.data, t_loss.data,
                                         np.mean(loss_hist), combined_loss.data))

        # Save the models
        if epoch % cfg.SAVE_PERIOD_IN_EPOCHS == 0:
            # run test on val set
            test_acc, test_loss = late_fusion_test(model, val_data_loader, criterion)
            val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f,  Val Accuracy: %.3f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step, test_loss, test_acc))

            test_acc, test_loss = late_fusion_test(model, train_data_loader, criterion)
            train_acc_loss_info.write('Epoch [%d/%d], Step [%d/%d], Train Loss: %.7f, Train Accuracy: %.3f \n'
                                      % (epoch, cfg.EPOCH_COUNT, i, total_step, test_loss, test_acc))

            torch.save(model.state_dict(), os.path.join(cfg.TRAIN_MODEL_PATH, 'spatial_model-%d.pkl' % epoch))


def early_fusion_train_epoch(train_data_loader, val_data_loader, criterion, loss_hist, train_loss_info, val_loss_info, train_acc_loss_info, pretrained):
    model = FusedModel(cfg.EARLY, pretrained=pretrained)

    # use GPU if available.
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, list(model.parameters())),  lr=cfg.LEARNING_RATE)
    # train the model
    total_step = len(train_data_loader)

    test_acc, test_loss = early_fusion_test(model, val_data_loader, criterion)
    val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f, Val Accuracy: %.3f \n'
                        % (0, cfg.EPOCH_COUNT, 0, total_step, test_loss, test_acc))

    test_acc, test_loss = early_fusion_test(model, train_data_loader, criterion)
    train_acc_loss_info.write('Epoch [%d/%d], Step [%d/%d], Train Loss: %.7f, Train Accuracy: %.3f \n'
                              % (0, cfg.EPOCH_COUNT, 0, total_step, test_loss, test_acc))


    for epoch in range(1, cfg.EPOCH_COUNT + 1):
        for i, (images, flows, labels) in enumerate(train_data_loader):
            # Set mini-batch images
            images = to_var(images, volatile=True)
            # Set mini-batch optical flows
            flows = to_var(flows, volatile=True)
            # Set mini-batch ground truth
            labels = to_var_labels(labels, volatile=False)

            # zero the parameter gradients
            # s_optimizer.zero_grad()
            # t_optimizer.zero_grad()

            # Forward, Backward and Optimize
            model.spatial_model.zero_grad()
            model.temporal_model.zero_grad()

            # feed images to CNN model
            rgb_out = model.spatial_model(images)
            flow_out = model.temporal_model(flows)
            final_logit = model.fuse(rgb_out, flow_out)

            combined_loss = criterion(final_logit, labels)

            combined_loss.backward()
            optimizer.step()

            loss_hist.append(float(combined_loss))

            # Print log info
            if i % cfg.LOG_STEP == 0:
                print(
                    'Epoch [%d/%d], Step [%d/%d], Early Fusion Combined Loss: %.7f, Running Loss: %5.6f, Combined Loss: %.7f'
                    % (epoch, cfg.EPOCH_COUNT, i, total_step, combined_loss.data, np.mean(loss_hist),
                       combined_loss.data))

                train_loss_info.write(
                    'Epoch [%d/%d], Step [%d/%d], Combined Loss: %.7f, Running Loss: %5.6f, Combined Loss: %.7f \n'
                    % (epoch, cfg.EPOCH_COUNT, i, total_step, combined_loss.data,
                       np.mean(loss_hist), combined_loss.data))

        # Save the models
        if epoch % cfg.SAVE_PERIOD_IN_EPOCHS == 0:
            # run test on val set
            test_acc, test_loss = early_fusion_test(model, val_data_loader, criterion)
            val_loss_info.write('Epoch [%d/%d], Step [%d/%d], Val Loss: %.7f, Val Accuracy: %.3f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step, test_loss, test_acc))

            test_acc, test_loss = early_fusion_test(model, train_data_loader, criterion)
            train_acc_loss_info.write('Epoch [%d/%d], Step [%d/%d], Train Loss: %.7f, Train Accuracy: %.3f \n'
                                % (epoch, cfg.EPOCH_COUNT, i, total_step, test_loss, test_acc))

            torch.save(model.state_dict(), os.path.join(cfg.TRAIN_MODEL_PATH, 'spatial_model-%d.pkl' % epoch))


def late_fusion_test(model, data_loader, criterion):
    test_loss = 0,
    correct = 0
    ground_truths, predicteds = [], []
    model.eval()

    with torch.no_grad():
        for i, (images, flows, labels) in enumerate(data_loader):
            # Set mini-batch images
            images = to_var(images, volatile=True)
            # Set mini-batch optical flows
            flows = to_var(flows, volatile=True)
            # Set mini-batch ground truth
            labels = to_var_labels(labels, volatile=False)

            for j in labels.tolist():
                ground_truths.append(j)

            s_predicted_labels = model.spatial_model(images)
            t_predicted_labels = model.temporal_model(flows)

            s_loss = criterion(s_predicted_labels, labels)
            t_loss = criterion(t_predicted_labels, labels)

            combined_loss = s_loss + t_loss
            test_loss += combined_loss.cpu().data.numpy()  # sum up batch loss

            s_outputs = F.softmax(s_predicted_labels)
            t_outputs = F.softmax(t_predicted_labels)
            outputs = (s_outputs + t_outputs) / 2.0

            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

            for j in pred.tolist():
                predicteds.append(j[0])

    test_loss /= len(data_loader)

    print(test_loss)

    score = metrics.accuracy_score(ground_truths, predicteds)
    cls_report = metrics.classification_report(ground_truths, predicteds)
    conf_mat = metrics.confusion_matrix(ground_truths, predicteds)
    print("Accuracy of Spatial = " + str(score))
    test_acc = score * 100.0
    print(test_acc)

    print(cls_report)
    print(conf_mat)

    return test_acc, test_loss


def early_fusion_test(model, data_loader, criterion):
    test_loss = 0,
    correct = 0
    ground_truths, predicteds = [], []
    model.eval()

    with torch.no_grad():
        for i, (images, flows, labels) in enumerate(data_loader):
            # Set mini-batch images
            images = to_var(images, volatile=True)
            # Set mini-batch optical flows
            flows = to_var(flows, volatile=True)
            # Set mini-batch ground truth
            labels = to_var_labels(labels, volatile=False)

            for j in labels.tolist():
                ground_truths.append(j)

            rgb_out = model.spatial_model(images)
            flow_out = model.temporal_model(flows)
            final_logit = model.fuse(rgb_out, flow_out)

            outputs = F.softmax(final_logit)

            loss = criterion(outputs, labels)
            test_loss += loss.cpu().data.numpy()  # sum up batch loss

            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

            for j in pred.tolist():
                predicteds.append(j[0])

    test_loss /= len(data_loader)
    # test_acc = 100. * correct / len(data_loader)
    # print(test_acc)
    print(test_loss)

    score = metrics.accuracy_score(ground_truths, predicteds)
    cls_report = metrics.classification_report(ground_truths, predicteds)
    conf_mat = metrics.confusion_matrix(ground_truths, predicteds)
    print("Accuracy of Spatial = " + str(score))
    test_acc = score * 100.0
    print(test_acc)
    print(cls_report)
    print(conf_mat)

    return test_acc, test_loss


def run_test(model_name, pretrained):
    set_name = 'test'
    img_path_list, flow_path_list, label_list = load_fused_dataset(set_name)

    # build data loader
    data_loader = get_fused_loader(img_path_list, flow_path_list, label_list, 1, shuffle=False,
                                   transform=VAL_TRANSFORM, num_workers=1)
    criterion = nn.CrossEntropyLoss()

    model = load_fusion_model(model_name, 20, pretrained)

    if model_name == 'late_fusion':
        late_fusion_test(model, data_loader, criterion)
    else:
        early_fusion_test(model, data_loader, criterion)

    # for model_name in ["late_fusion", "early_fusion"]:
    #     for set_name in ["train", "test"]:
    #
    #         img_path_list, flow_path_list, label_list = load_fused_dataset(set_name)
    #
    #         # build data loader
    #         data_loader = get_fused_loader(img_path_list, flow_path_list, label_list, 1, shuffle=False,
    #                                        transform=VAL_TRANSFORM, num_workers=1)
    #
    #         info = open(cfg.TRAIN_MODEL_PATH + '/' + model_name + "_" + set_name + '.txt', 'w')
    #
    #         for i in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
    #             criterion = nn.CrossEntropyLoss()
    #
    #             model = load_fusion_model(model_name, i, pretrained, "model/pretrained_" + model_name + "_1")
    #
    #             if model_name == 'late_fusion':
    #                 test_acc, test_loss = late_fusion_test(model, data_loader, criterion)
    #             else:
    #                 test_acc, test_loss = early_fusion_test(model, data_loader, criterion)
    #
    #             info.write('Loss: %.7f, Accuracy: %.3f \n'
    #                                 % (test_loss, test_acc))
