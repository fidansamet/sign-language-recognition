import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
import numpy as np
import config as cfg
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SpatialDataset(data.Dataset):
    """Spatial Model"""
    def __init__(self, img_name_list, img_label_list, transform=None):
        self.img_name_list = img_name_list
        self.img_label_list = img_label_list
        self.transform = transform

    def __getitem__(self, index):
        # returns one data pair (image and label)
        try:
            img = Image.open(self.img_name_list[index])
            img = img.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            img_label = self.img_label_list[index]

        except Exception as e:
            print("Image I/O error!", e)
            pass

        return img, img_label

    def __len__(self):
        return len(self.img_name_list)


def get_spatial_loader(img_name_list, img_label_list, batch_size, shuffle, transform, num_workers):
    dataset = SpatialDataset(img_name_list=img_name_list, img_label_list=img_label_list, transform=transform)

    # returns (images, labels) for every iteration
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    return data_loader


class TemporalDataset(data.Dataset):
    """Temporal Model"""
    def __init__(self, flow_name_list, flow_label_list, transform=None):
        self.flow_name_list = flow_name_list
        self.flow_label_list = flow_label_list
        self.transform = transform

    def __getitem__(self, index):
        # returns one data pair (array and label)
        try:
            optical_flow = []
            for i in range(index, index + 10):
                cur_flow = np.load(self.flow_name_list[index])
                if i == index:
                    optical_flow = cur_flow.f.arr_0
                else:
                    # concatenate row-wise
                    optical_flow = np.concatenate((optical_flow, cur_flow.f.arr_0), axis=2)

            optical_flow = self.crop_center(optical_flow, cfg.MIN_RESIZE, cfg.MIN_RESIZE)
            flow_label = self.flow_label_list[index]

        except Exception as e:
            print("npz I/O error!", e)
            pass

        return np.einsum('ijk->kij', optical_flow), flow_label

    def __len__(self):
        return len(self.flow_name_list)

    def crop_center(self, arr, cropx, cropy):
        y, x, z = arr.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return arr[starty:starty + cropy, startx:startx + cropx]


def get_temporal_loader(flow_name_list, flow_label_list, batch_size, shuffle, transform, num_workers):
    dataset = TemporalDataset(flow_name_list=flow_name_list, flow_label_list=flow_label_list, transform=transform)

    # returns (flows, labels) for every iteration
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    return data_loader


class FusionDataset(data.Dataset):
    """Fusion Model"""
    def __init__(self, img_name_list, flow_name_list, label_list, transform=None):
        self.img_name_list = img_name_list
        self.flow_name_list = flow_name_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, index):
        # returns one data pair (flow, image and label)
        try:
            optical_flow = []
            for i in range(index, index + 10):
                cur_flow = np.load(self.flow_name_list[index])
                if i == index:
                    optical_flow = cur_flow.f.arr_0
                else:
                    # concatenate row-wise
                    optical_flow = np.concatenate((optical_flow, cur_flow.f.arr_0), axis=2)

            optical_flow = self.crop_center(optical_flow, cfg.MIN_RESIZE, cfg.MIN_RESIZE)

            img = Image.open(self.img_name_list[index])
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            label = self.label_list[index]

        except Exception as e:
            print("npz I/O error!", e)
            pass

        return img, np.einsum('ijk->kij', optical_flow), label

    def __len__(self):
        return len(self.flow_name_list)

    def crop_center(self, arr, cropx, cropy):
        y, x, z = arr.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return arr[starty:starty + cropy, startx:startx + cropx]


def get_fused_loader(img_name_list, flow_name_list, label_list, batch_size, shuffle, transform, num_workers):
    dataset = FusionDataset(img_name_list=img_name_list, flow_name_list=flow_name_list,
                            label_list=label_list, transform=transform)

    # returns (images, flows, labels) for every iteration
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
    return data_loader
