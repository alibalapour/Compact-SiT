import os
import gdown
import shutil
import gdown
import zipfile

from torchvision import transforms
from torchvision.datasets import ImageFolder

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from datasets.TinyImageNet import TinyImageNetDataset
from datasets.CIFAR import CIFAR10, CIFAR100
from datasets.STL10 import STL10
from datasets.UH import UHDataset
from datasets.NCT import NCTDataset
from datasets.BreakHis import BreakHis
from datasets.MHIST import MHIST
from datasets.CustomDataset import CustomDataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(os.path.join(args.dataset_location, 'CIFAR10_dataset'),
                          download=True, train=is_train, transform=transform,
                          num_imgs_per_cat=args.num_imgs_per_cat,
                          training_mode=args.training_mode)
        nb_classes = 10


    elif args.dataset == 'CIFAR100':
        dataset = CIFAR100(os.path.join(args.dataset_location, 'CIFAR100_dataset'),
                           download=True, train=is_train, transform=transform,
                           num_imgs_per_cat=args.num_imgs_per_cat,
                           training_mode=args.training_mode)

        nb_classes = 100


    elif args.dataset == 'STL10':
        #### Note num_imgs_per_cat is not implemented in this dataset as it has unlabeled data
        split = 'train+unlabeled' if args.training_mode == 'SSL' else 'train'
        split = split if is_train else 'test'

        dataset = STL10(root=os.path.join(args.dataset_location, 'STL10'),
                        download=True, split=split, transform=transform,
                        training_mode=args.training_mode)
        nb_classes = 10

    elif args.dataset == 'TinyImageNet':
        mode = 'train' if is_train else 'val'
        root_dir = os.path.join(args.dataset_location, 'TinyImageNet/tiny-imagenet-200/')
        dataset = TinyImageNetDataset(root_dir=root_dir, download=True, mode=mode, transform=transform,
                                      num_imgs_per_cat=args.num_imgs_per_cat,
                                      training_mode=args.training_mode)
        nb_classes = 200

    elif args.dataset == 'UH_mini':  # Mini Dataset for SSL  - ~120k
        if is_train:
            path = os.path.join(args.dataset_location, 'UH_mini_dataset')
            gdown.download(id='1-30No_EN3ISKvvAgmPq2PJerxaJOgMFb')
            # !gdown --id 1-30No_EN3ISKvvAgmPq2PJerxaJOgMFb
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./mini_dataset.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./mini_dataset.zip
            shutil.move('./content/datasets', path)
            shutil.rmtree('./content')
            os.remove('./mini_dataset.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = UHDataset(dataset_folder, transform)
        else:
            path = os.path.join(args.dataset_location, 'UH_val_dataset')
            gdown.download(id='1hMj_1jZIdDzfg75My7CPTvFvYiHdvdJH')
            # !gdown --id 1hMj_1jZIdDzfg75My7CPTvFvYiHdvdJH
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./val_dataset.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./val_dataset.zip
            shutil.move('./content/datasets', path)
            shutil.rmtree('./content')
            os.remove('./val_dataset.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = UHDataset(dataset_folder, transform)
        nb_classes = 1

    elif args.dataset == 'UH_main':  # Main Dataset for SSL  - ~600k
        if is_train:
            path = os.path.join(args.dataset_location, 'UH_main')
            gdown.download(id='1Cipc0LflqReenVwPrAldZ3ZR_11fLy5m')
            # !gdown --id 1JoJxnY4zPuvjVGIALE_UCpCX97i6HA8J
            # !gdown --id 1Cipc0LflqReenVwPrAldZ3ZR_11fLy5m
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./dataset_v1.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./dataset_v1.zip
            shutil.move('./content/datasets', path)
            shutil.rmtree('./content')
            os.remove('./dataset_v1.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = UHDataset(dataset_folder, transform)
        else:
            path = os.path.join(args.dataset_location, 'UH_val_dataset')
            gdown.download(id='1hMj_1jZIdDzfg75My7CPTvFvYiHdvdJH')
            # !gdown --id 1hMj_1jZIdDzfg75My7CPTvFvYiHdvdJH
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./val_dataset.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./val_dataset.zip
            shutil.move('./content/datasets', path)
            shutil.rmtree('./content')
            os.remove('./val_dataset.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = UHDataset(dataset_folder, transform)
        nb_classes = 1

    elif args.dataset == 'NCT':
        if is_train:
            transform = build_transform(is_train, args)
            dataset_folder = ImageFolder('../input/nct-dataset/NCT_dataset/NCT-CRC-HE-100K')
            #             dataset_folder = ImageFolder('../input/dataset-nct/NCT_dataset/NCT-CRC-HE-100K')
            dataset = NCTDataset(dataset_folder, transform, args.training_mode)
        else:
            transform = build_transform(is_train, args)
            dataset_folder = ImageFolder('../input/nct-test-dataset/CRC-VAL-HE-7K')
            dataset = NCTDataset(dataset_folder, transform, args.training_mode)
        nb_classes = 9

    elif args.dataset == 'BreakHis':
        pass
        # if is_train:
        #     transform = build_transform(is_train, args)
        #     dataset_folder = ImageFolder('../input/breakhis/BreakHis/content/BreakHis')
        #     dataset = BreakHis(dataset_folder, transform, args.training_mode)
        # else:
        #     transform = build_transform(is_train, args)
        #     dataset_folder = ImageFolder('../input/breakhis/BreakHis_test/content/BreakHis_test')
        #     dataset = BreakHis(dataset_folder, transform, args.training_mode)
        # nb_classes = 2

    elif args.dataset == 'BreakHis_2':
        if is_train:
            path = os.path.join(args.dataset_location, 'BreakHis')
            gdown.download(id='1xzaFz4isBsWCP0U0dI-uSy9yVH9owWcz')
            # !gdown --id 1xzaFz4isBsWCP0U0dI-uSy9yVH9owWcz
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./BreakHis.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./BreakHis.zip
            shutil.move('./content/BreakHis', path)
            shutil.rmtree('./content')
            os.remove('./BreakHis.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = BreakHis(dataset_folder, transform)
        else:
            path = os.path.join(args.dataset_location, 'BreakHis_test')
            gdown.download(id='1-0H_y_DlbWS0T5k_GfKma1MAiZSRsOTC')
            # !gdown --id 1-0H_y_DlbWS0T5k_GfKma1MAiZSRsOTC
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./BreakHis_test.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./BreakHis_test.zip
            shutil.move('./content/BreakHis_test', path)
            shutil.rmtree('./content')
            os.remove('./BreakHis_test.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = BreakHis(dataset_folder, transform)
        nb_classes = 2

    elif args.dataset == 'MHIST':
        if is_train:
            path = os.path.join(args.dataset_location, 'MHIST')
            gdown.download(id='1-eOvOO2y4VdKOM70whYUHBRk0EWqPatz')
            # !gdown --id 1-eOvOO2y4VdKOM70whYUHBRk0EWqPatz
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./MHIST.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./MHIST.zip
            shutil.move('./content/MHIST', path)
            shutil.rmtree('./content')
            os.remove('./MHIST.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = MHIST(dataset_folder, transform)
        else:
            path = os.path.join(args.dataset_location, 'MHIST_test')
            gdown.download(id='1-hdXBwJPAi-SJEQbHXZvpvrENFGfiyqx')
            # !gdown --id 1-hdXBwJPAi-SJEQbHXZvpvrENFGfiyqx
            shutil.rmtree(path, ignore_errors=True)
            with zipfile.ZipFile('./MHIST_test.zip', 'r') as zipObj:
                zipObj.extractall()
            # !unzip -qq ./MHIST_test.zip
            shutil.move('./content/MHIST_test', path)
            shutil.rmtree('./content')
            os.remove('./MHIST_test.zip')
            transform = build_transform(is_train=is_train, args=args)
            dataset_folder = ImageFolder(path)
            dataset = MHIST(dataset_folder, transform)
        nb_classes = 2

    elif args.dataset is None:
        if is_train:
            path = os.path.join(args.dataset_location, 'Custom_Train')
            transform = build_transform(is_train=is_train, args=args)
            try:
                dataset_folder = ImageFolder(path)
                dataset = CustomDataset(dataset_folder, transform)
            except:
                raise ValueError('your custom train dataset has probelm')
        else:
            path = os.path.join(args.dataset_location, 'Custom_Test')
            transform = build_transform(is_train=is_train, args=args)
            try:
                dataset_folder = ImageFolder(path)
                dataset = CustomDataset(dataset_folder, transform)
            except:
                raise ValueError('your custom test dataset has probelm')
        nb_classes = len(dataset_folder.classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
