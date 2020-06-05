import torch
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import config as cfg


class TargetDataSet(data.Dataset):
    def __init__(self, im_names_list, label_list, transform=None):
        'Initialization'
        self.im_names_list = im_names_list
        self.transform = transform
        self.label_list = label_list

    def __getitem__(self, index):
        """Returns one data pair (image and label)."""
        'Generates data of batch_size samples'
        try:
            img = Image.open(self.im_names_list[index])
            img = img.convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            label = self.label_list[index]

        except Exception as e:
            print("Image I/O error!", e)
            pass

        return img, label

    def __len__(self):
        return len(self.im_names_list)


def get_loader(im_names_list, label_list, batch_size, shuffle, transform, num_workers):
    # type: (object, object, object, object, object, object) -> object
    """Returns torch.utils.data.DataLoader for custom pascal dataset."""
    # PASCAL  dataset
    dream_data = TargetDataSet(im_names_list=im_names_list,
                               label_list=label_list,
                               transform=transform)

    # Data loader for Custom dataset
    # This will return (images, labels) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # labels: list indicating label info for each image.
    data_loader = torch.utils.data.DataLoader(dataset=dream_data,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader
