# import os
# from os import listdir
# import numpy as np
# from os.path import isfile, join
# import config as cfg
# import copy
#
#
# def prep_train_data(main_path):
#
#     files_all = []
#     files_temp = []
#     files_cls = []
#     labels_all = []
#     labels_temp = []
#     # get all folders for each class
#     test_ff = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
#     test_f = sorted(test_ff)
#     # test_f = [test_folders[0] for test_folders in os.walk(main_path)][1:]
#     # now save test_f to a file for later usage
#     with open("./class_dict.txt", 'w') as f:
#         for i, cls in enumerate(test_f):
#             f.write(cls + '\n')
#     # get files for each folder
#     for ii, folder in enumerate(test_f):
#         index_ = ii
#         test_files = [files_cls.append(os.path.join(main_path, folder, f)) for f in listdir(os.path.join(main_path, folder)) if isfile(join(main_path, folder, f))]
#
#         with open(os.path.join(main_path, folder) + ".txt", 'w') as ff:
#             for aa in files_cls:
#                 files_all.append(aa)
#                 ff.write(folder + "/" + os.path.basename(aa) + '\n')
#
#         files_cls = []
#         labels_temp.append(np.zeros((len(test_files),), dtype=int)[:] + index_)
#
#     for lst in labels_temp:
#         for i in lst:
#             labels_all.append(i)
#
#     return files_all, labels_all, test_f
#
#
# def prep_test_data(main_path):
#
#     test_files_all = []
#     test_labels_all = []
#     test_labels_temp = []
#     # get all folders for each class
#     test_f = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
#     # read class list
#     with open("./class_dict.txt", 'r') as f:
#         cls_list = f.read().splitlines()
#     # test_f = [test_folders[0] for test_folders in os.walk(main_path)][1:]
#     # get files for each folder
#     for folder in test_f:
#         index_ = cls_list.index(folder)
#         test_files = [test_files_all.append(os.path.join(main_path, folder, f)) for f in listdir(os.path.join(main_path, folder)) if isfile(join(main_path, folder, f))]
#         test_labels_temp.append(np.zeros((len(test_files),),dtype=int)[:] + index_)
#
#     for lst in test_labels_temp:
#         for i in lst:
#             test_labels_all.append(i)
#
#     return test_files_all, test_labels_all, cls_list
#
#
# # def prep_test_data_MC(main_path):
# #
# #     test_files_all = []
# #     test_labels_all = []
# #     test_labels_temp = []
# #     # get all folders for each class
# #     test_f = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
# #     # read class list
# #     with open("./class_dict.txt", 'r') as f:
# #         cls_list = f.read().splitlines()
# #     # test_f = [test_folders[0] for test_folders in os.walk(main_path)][1:]
# #     # get files for each folder
# #     for folder in test_f:
# #         index_ = cls_list.index(folder)
# #         test_files = [test_files_all.append(os.path.join(main_path, folder, f)) for f in listdir(os.path.join(main_path, folder)) if isfile(join(main_path, folder, f))]
# #         test_labels_temp.append(np.zeros((len(test_files),),dtype=int)[:] + index_)
# #
# #         # fix labels
# #         b = []
# #         for ss, s2 in enumerate(cfg.MC_CLASS_IDS):
# #             if s2 in folder:
# #                 b.append(ss)
# #         labels_temp = [b] * len(test_files)
# #         test_labels_all.extend(copy.deepcopy(labels_temp))
# #
# #     return test_files_all, test_labels_all, cls_list
# #
# #
# # def prep_train_data_MC(main_path):
# #
# #     files_all = []
# #     files_cls = []
# #     labels_all = []
# #     labels_temp = []
# #     # get all folders for each class
# #     test_ff = [d for d in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, d))]
# #     test_f = sorted(test_ff)
# #     # now save test_f to a file for later usage
# #     with open("./class_dict.txt", 'w') as f:
# #         for i, cls in enumerate(test_f):
# #             f.write(cls + '\n')
# #     # get files for each folder
# #     for ii, folder in enumerate(test_f):
# #         test_files = [files_cls.append(os.path.join(main_path, folder, f)) for f in listdir(os.path.join(main_path, folder)) if isfile(join(main_path, folder, f))]
# #
# #         with open(os.path.join(main_path, folder) + ".txt", 'w') as ff:
# #             for aa in files_cls:
# #                 files_all.append(aa)
# #                 ff.write(folder + "/" + os.path.basename(aa) + '\n')
# #
# #         files_cls = []
# #         # fix labels
# #         b = []
# #         for ss, s2 in enumerate(cfg.MC_CLASS_IDS):
# #             if s2 in folder:
# #                 b.append(ss)
# #         labels_temp = [b] * len(test_files)
# #         labels_all.extend(copy.deepcopy(labels_temp))
# #         # labels_temp.append(np.zeros((len(test_files), len(cfg.MC_CLASS_IDS)), dtype=int)[:] + index_)
# #
# #     return files_all, labels_all, test_f
#
#
# #
# # def load_data_set_train(dataset_path="./datasets_train/dataset_somera/"):
# #     # image path lerinin oldugu dosya
# #     with open(dataset_path + "file_names.txt") as f:
# #         image_path_list = f.read().splitlines()
# #     # image lar ile ayni sirada olacak sekilde label bilgisi
# #     with open(dataset_path + "labels.txt") as f:
# #         labels_list = f.read().splitlines()
# #
# #     return image_path_list, labels_list
# #
# #
# # def load_data_set_test(test_path):
# #     # image path lerinin oldugu dosya
# #     with open(test_path + "file_names_test.txt") as f:
# #         image_path_list = f.read().splitlines()
# #     # image lar ile ayni sirada olacak sekilde label bilgisi
# #     with open(test_path + "labels_test.txt") as f:
# #         labels_list = f.read().splitlines()
# #
# #     return image_path_list, labels_list