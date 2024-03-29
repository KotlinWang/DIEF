import os
import random
import glob
import json
import torch
import torchvision
from sklearn import metrics
import warnings
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore")

normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


class MakeJson(object):
    """
    制作json文件
    将文件的路径保存到train、test的json文件中
    """

    def __init__(self, config):
        self.data_dir = config.data_dir
        self.class_list = config.class_list
        self.class_dict = config.class_dict
        self.train_json_dir = config.train_json_dir
        self.test_json_dir = config.test_json_dir
        self.split_rate = config.split_rate

        train_list = []
        test_list = []
        data_dic = self._get_data_path()
        for label_name in self.class_list:  # 获取数据类别
            split_flag = int(len(data_dic[label_name]) * self.split_rate)
            random.shuffle(data_dic[label_name])
            train_list.extend((_, self.class_dict[label_name]) for _ in data_dic[label_name][:split_flag])
            test_list.extend((_, self.class_dict[label_name]) for _ in data_dic[label_name][split_flag:])
        random.shuffle(train_list)
        random.shuffle(test_list)
        self._to_json(save_list=train_list, save_dir=self.train_json_dir)
        self._to_json(save_list=test_list, save_dir=self.test_json_dir)

    def _get_data_path(self):
        files = os.listdir(self.data_dir)
        data_dic = {}
        for file_name in files:
            data_dic[file_name] = glob.glob(os.path.join(self.data_dir, file_name) + '/*.txt')
        return data_dic

    def _to_json(self, save_list, save_dir):
        test_dict = {
            'version': "1.0",
            'results': save_list,
        }
        json_str = json.dumps(test_dict, indent=0)
        with open(save_dir, 'w') as json_file:
            json_file.write(json_str)
        print('Finish...')


class SimpleLoss(torch.nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, ypred, ytgt):
        # ypred = torch.max(ypred.data, 1)[1]
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_metrics(preds, labels):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        max_labels = torch.max(labels.data, 1)[1].cpu()
        max_preds = torch.max(preds.data, 1)[1].cpu()

        acc = metrics.accuracy_score(max_labels, max_preds)
        precision = metrics.precision_score(max_labels, max_preds, average='macro')
        f1_score = metrics.f1_score(max_labels, max_preds, average='macro')
        recall = metrics.recall_score(max_labels, max_preds, average='macro')
        auc = metrics.roc_auc_score(max_labels, max_preds)

    return acc, precision, f1_score, recall, auc


def get_val_info(cfg, model, valloader, loss_fn, device):
    model.eval()
    counter = 0
    total_loss = 0.0
    total_acc = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    total_recall = 0.0
    total_auc = 0.0

    hidden = None
    # print('running eval...')
    with torch.no_grad():
        for audios, images, labels in tqdm(valloader):
            counter += 1
            labels = torch.zeros(len(images), cfg['num_classes']).scatter_(1, labels.view(len(labels), -1), 1).to(device)
            audios = audios.to(device)
            images = images.to(device)
            preds, _ = model(
                audios,
                images,
                hidden
            )

            # loss
            total_loss += loss_fn(preds, labels).item() * preds.shape[0]

            # iou
            # labels = torch.max(labels.data, 1)[1].cpu()
            # preds = torch.max(preds.data, 1)[1].detach().cpu()
            acc, precision, f1_score, recall, auc = get_metrics(preds, labels)
            total_acc += acc
            total_precision += precision
            total_f1 += f1_score
            total_recall += recall
            total_auc += auc

    model.train()
    return {
        'loss': total_loss / len(valloader.dataset),
        'acc': total_acc / counter,
        'precision': total_precision / counter,
        'f1_score': total_f1 / counter,
        'recall': total_recall / counter,
        'auc': total_auc / counter
    }


def test_result(cfg, model, valloader, device):
    model.load_state_dict(torch.load(cfg['model_dir'], map_location=device))
    model.eval()
    true_list = []
    pred_list = []
    with torch.no_grad():
        for audios, images, labels in valloader:
            # labels = torch.zeros(len(images), cfg['num_classes']).scatter_(1, labels.view(len(labels), -1), 1).to(device)
            audios = audios.to(device)
            images = images.to(device)
            preds, _ = model(
                audios,
                images,
                None
            )
            # labels = torch.max(labels.data, 1)[1].cpu()
            # preds = torch.max(preds.data, 1)[1].detach().cpu()
            preds = preds.data.cpu()
            true_list.extend(labels.tolist())
            pred_list.extend(np.argmax(preds, axis=1).tolist())

    return true_list, pred_list
