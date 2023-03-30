from data import make_dataset, ImageLabelFilelist, default_flist_reader, ImageFolderWithLabels
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import sys


def main():
    crop = True
    train = False
    # images = sorted(make_dataset('../from_server/anh/train/DG-Net/StanfordCars'))
    # print(images)
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((256, 256))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(256)] + transform_list if 256 is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolderWithLabels('../from_server/anh/train/DG-Net/StanfordCars', transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=8)
    test_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=8)
    for it, ([images_a, label_a], [images_b, label_b]) in enumerate(zip(loader, test_loader)):
        # print(it)
        print(label_a)
        print(label_b)
        sys.exit()
    # it = iter(loader)
    # first = next(it)
    # print(first)


if __name__ == '__main__':
    main()
