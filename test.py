import argparse
from ast import arg
import os
import csv

import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
import time
from datetime import datetime


from options.test_options import TestOptions

SEED = 0


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def test(model, loader, threshold=0.5):
    model.eval()
    y_pred = []
    ids = []
    with torch.no_grad():
        y_pred = []
        print("Length of dataset: %d" % (len(loader)))
        for img,label in loader:
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())

    y_pred = np.array(y_pred) > threshold

    # ================== save this if you want to plot the curves =========== #
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #



    # Acc based on 0.5


    # Acc based on the best thres


    return y_pred.astype(int)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeTestDataset(Dataset):
    """"data_mode = bdb because our team name is 不登榜"""
    def __init__(self, img_path,img_label,
                 arch,
                 jpeg_quality=None,
                 gaussian_sigma=None):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        # = = = = = = data path = = = = = = = = = #
        self.total_list = img_path
        # ------------data label ---------------- #
        self.img_label = img_label


        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]

        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        label = self.img_label[idx]
        return img, label


# 定义加载模型参数函数
def load_model(model, pretrained_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    # 筛选出不属于CLIP部分的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'clip' not in k}

    # 更新模型参数
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


if __name__ == '__main__':
    start_time = time.time()  # 开始计时
    """在options/test_options.py，author已经定义了test的参数"""
    opt = TestOptions().parse()
    # 用于存储结果csv文件
    if os.path.exists(opt.predict_path):
        shutil.rmtree(opt.predict_path)
    os.makedirs(opt.predict_path)

    model = get_model(opt.arch)  # get model network
    # pretrained_weights/fc_weights.pth
    state_dict = torch.load(opt.premodel_path, map_location='cpu')  # pretrained model weights
    model.load_state_dict(state_dict['model'])
    print("Model loaded..")
    model.eval()
    model.cuda()
    # # test A 榜
    # dataset_paths = [dict(test_dataset=opt.test_dataset_path)]  # key : test_dataset
    # test_label_path = "./datasets/testset.txt"
    # test_label = pd.read_csv(test_label_path)
    # test_label['path'] = "./datasets/faceA/" + test_label['img_name'].astype(str)
    # for dataset_path in (dataset_paths):
    #     set_seed()
    #     dataset = RealFakeTestDataset( test_label['path'],test_label['target'],
    #                         opt.arch,
    #                               jpeg_quality= None,
    #                               gaussian_sigma= None,
    #                               )
    #
    #     loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle=False, num_workers=0)
    #     prediction = test(model, loader)
    #     test_label['target'] = prediction
    #
    # # 输出CSV
    # test_label = test_label.sort_values(by='img_name', key=lambda col: col)
    # test_label[['img_name', 'target']].to_csv('cla_pre.csv', index=False,header=False)
    # print("运行完成")

    # 只使用 datasets/faceB/ 文件夹下的所有图片作为测试集
    test_dataset_path = "./datasets/faceB/"
    test_images = [os.path.join(test_dataset_path, img) for img in os.listdir(test_dataset_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_labels = [0] * len(test_images)  # 假设所有图片的初始标签为0

    set_seed()
    dataset = RealFakeTestDataset(test_images, test_labels, opt.arch, jpeg_quality=None, gaussian_sigma=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    prediction = test(model, loader)

    # 获取当前时间
    current_time = datetime.now().strftime('%m%d%H%M')  # 获取月日时分，格式如10172153
    file_name = f"{current_time}cla_pre.csv"

    # 输出预测结果到CSV文件
    output_df = pd.DataFrame({'img_name': [os.path.basename(path) for path in test_images], 'target': prediction})
    output_df = output_df.sort_values(by='img_name', key=lambda col: col)
    output_df.to_csv(f'./result/{file_name}', index=False, header=False)
    print("运行完成")
    end_time = time.time()  # 结束计时
    elapsed_time = end_time - start_time
    print(f"总运行时间: {elapsed_time:.2f} 秒")





