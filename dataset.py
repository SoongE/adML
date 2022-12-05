# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 23:23:51 2019
@author: Keshik
"""
import torchvision.datasets.voc as voc
from torchvision.transforms import transforms

try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from PIL import Image


class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, year='2012', image_set='train', download=False, basic_transform=None, pixel_transform=None,
                 channel_transform=None):
        super().__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,
            target_transform=None)
        self.basic_transform = basic_transform
        self.pixel_transform = pixel_transform
        self.channel_transform = channel_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        # target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        if self.pixel_transform is None or self.channel_transform is None:
            return self.basic_transform(img)

        return self.basic_transform(img), self.pixel_transform(img), self.channel_transform(img)

    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)


def get_transform():
    mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    basic_aug = transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.CenterCrop((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                    ])

    pos_aug = transforms.Compose([transforms.Resize((256, 256)),
                                  transforms.RandomCrop((224, 224), padding=4),
                                  transforms.RandomRotation(15),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=mean, std=std),
                                  ])

    channel_aug = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.CenterCrop((224, 224)),
                                      transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std),
                                      ])
    return basic_aug, pos_aug, channel_aug
