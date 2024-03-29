import os
import json
import random
import numpy as np
from PIL import Image
from python_speech_features import mfcc, delta

import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)


def get_mfcc(x, sr):
    wav_feature = mfcc(signal=x, samplerate=sr, winlen=0.04, winstep=0.008, numcep=13,
                       nfilt=40, nfft=2048, lowfreq=100, highfreq=None, preemph=1.25, ceplifter=22)
    d_mfcc_feat1 = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    wav_feature = np.hstack((wav_feature, d_mfcc_feat1, d_mfcc_feat2))

    # wav_feature = x.reshape(-1, 1)

    return wav_feature


def normalize(tensor):
    tensor_minusmean = tensor - tensor.mean()
    return tensor_minusmean / tensor_minusmean.max()


class GetDataIter(Data.Dataset):
    """自动加载数据"""

    def __init__(self, root, file_path, sr, device):
        self.root = root
        self.device = device
        self.file_path = file_path
        self.sr = sr
        with open(self.file_path, 'r') as file:
            self.file_list = file.readlines()
            random.shuffle(self.file_list)

    def __getitem__(self, index):
        # random.shuffle(self.file_list)
        audio = np.loadtxt(os.path.join(self.root, 'AudioBase', '{}.txt'.format(self.file_list[index].split(' ')[0])))
        img = Image.open(os.path.join(self.root, 'ImageBase', '{}.jpg'.format(self.file_list[index].split(' ')[0])))
        label = np.array(self.file_list[index].split(' ')[1], dtype=int)

        audio = normalize(get_mfcc(audio, sr=self.sr))
        audio = torch.FloatTensor(audio)
        img = transform(img)
        y = torch.tensor(label)

        return audio, img, y

    def __len__(self):
        return len(self.file_list)


def compile_data(cfg, device):
    train_data = GetDataIter(cfg['dataset'], os.path.join(cfg['dataset'], 'json/' + 'train.txt'), cfg['sr'], device)
    test_data = GetDataIter(cfg['dataset'], os.path.join(cfg['dataset'], 'json/' + 'test.txt'), cfg['sr'], device)
    train_loader = Data.DataLoader(
        dataset=train_data,  # torch TensorDataset format
        batch_size=cfg['batch_size'],  # mini batch size
        shuffle=cfg['is_shuffle'],  # 要不要打乱数据 (打乱比较好)
        num_workers=cfg['num_workers'],  # 多线程来读数据
        drop_last=cfg['drop_last'],  # 自动舍弃最后不足batchsize的batch
        pin_memory=True
    )
    test_loader = Data.DataLoader(
        dataset=test_data,  # torch TensorDataset format
        batch_size=cfg['batch_size'],  # mini batch size
        shuffle=cfg['is_shuffle'],  # 要不要打乱数据 (打乱比较好)
        num_workers=cfg['num_workers'],  # 多线程来读数据
        drop_last=cfg['drop_last'],  # 自动舍弃最后不足batchsize的batch
        pin_memory=True
    )
    return train_loader, test_loader
