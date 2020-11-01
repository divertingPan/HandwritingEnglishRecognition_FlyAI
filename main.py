# -*- coding: utf-8 -*-
import argparse
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from torch.optim import Adam, Adadelta, RMSprop, AdamW
from torchvision.transforms import transforms
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
from flyai.utils.log_helper import train_log
from path import MODEL_PATH, DATA_PATH, DATA_ID
import crnn
import utils
import dataset
import sys
import matplotlib.pyplot as plt
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=50, type=int, help="batch size")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alphabet = "abcdefghijklmnopqrstuvwxyz-' "

# random.seed(777)
np.random.seed(777)
torch.manual_seed(777)


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Main(FlyAI):
    """
    项目中必须继承FlyAI类，否则线上运行会报错。
    """

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids(DATA_ID)

    def deal_with_data(self):
        """
        处理数据，没有可不写。
        :return:
        """
        pass

    def train(self):
        """
        训练模型，必须实现此方法
        :return:
        """

        print('load csv from %s' % os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
        df = pd.read_csv(os.path.join(DATA_PATH, DATA_ID, 'train.csv'))
        # image_path_list = df['image_path'].values
        image_path_list = df['image_path'].values
        label_list = df['label'].values

        # 划分训练集和校验集
        all_size = len(image_path_list)
        train_size = int(all_size * 0.8)
        train_image_path_list = image_path_list[:train_size]
        train_label_list = label_list[:train_size]
        val_image_path_list = image_path_list[train_size:]
        val_label_list = label_list[train_size:]
        print('train_size: %d, val_size: %d' % (len(train_image_path_list), len(val_image_path_list)))

        train_transform = transforms.Compose([
            transforms.RandomAffine(degrees=3, shear=0.1, fillcolor=255)
        ])
        # train_transform = None

        train_data = dataset.OCRDataset(img_path=train_image_path_list, label_value=train_label_list,
                                        transform=train_transform)
        val_data = dataset.OCRDataset(img_path=val_image_path_list, label_value=val_label_list, mode='val')
        train_loader = DataLoader(train_data, batch_size=args.BATCH, shuffle=True, drop_last=True,
                                  collate_fn=dataset.alignCollate(imgH=32, imgW=100, keep_ratio=True))
        val_loader = DataLoader(val_data, batch_size=args.BATCH, shuffle=True, drop_last=True,
                                collate_fn=dataset.alignCollate(imgH=32, imgW=100, keep_ratio=True))

        net = crnn.CRNN(imgH=32, nc=1, nclass=len(alphabet) + 1, nh=256, leakyRelu=True).to(device)
        net.apply(weights_init)

        converter = utils.strLabelConverter(alphabet)
        loss_fn = CTCLoss()

        # setup optimizer
        # optimizer = Adam(net.parameters(), lr=0.001)
        # optimizer = AdamW(net.parameters(), lr=0.001)
        optimizer = Adadelta(net.parameters())
        # optimizer = RMSprop(net.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)

        loss_avg = utils.averager()

        best = 0
        for epoch in range(args.EPOCHS):
            for step, (cpu_images, cpu_texts) in enumerate(train_loader):

                # img = np.asarray(cpu_images[0]).transpose((1, 2, 0))
                # plt.imshow(img)
                # plt.show()

                net.train()
                text, length = converter.encode(cpu_texts)
                # print(cpu_images.dtype)
                preds = net(cpu_images.to(device))
                preds_size = torch.IntTensor([preds.size(0)] * args.BATCH)
                cost = loss_fn(preds, text, preds_size, length)
                net.zero_grad()
                cost.backward()
                optimizer.step()
                loss_avg.add(cost / args.BATCH)

                if step % 200 == 0:
                    print('[epoch %d / total %d][step %d / total %d] Loss: %f -- lr:%f' %
                          (epoch+1, args.EPOCHS, step, len(train_loader), loss_avg.val(), optimizer.param_groups[0]['lr']))
                    loss_avg.reset()

                    _, preds = preds.max(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                    print(sim_preds)
                    print(cpu_texts)

                    # torch.cuda.empty_cache()

            scheduler.step()

            n_correct = 0
            net.eval()
            with torch.no_grad():
                for step, (cpu_images, cpu_texts) in enumerate(val_loader):

                    text, length = converter.encode(cpu_texts)
                    preds = net(cpu_images.to(device))
                    preds_size = torch.IntTensor([preds.size(0)] * args.BATCH)
                    cost = loss_fn(preds, text, preds_size, length)
                    loss_avg.add(cost / args.BATCH)
                    _, preds = preds.max(2)
                    # preds = preds.squeeze(2)
                    preds = preds.transpose(1, 0).contiguous().view(-1)
                    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
                    for pred, target in zip(sim_preds, cpu_texts):
                        if pred == target.lower():
                            n_correct += 1

                raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:10]
                for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
                    print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

                accuracy = n_correct / float(len(val_loader) * args.BATCH)
                print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

            # do checkpointing
            if accuracy > best:
                print('Best accuracy: ', accuracy, '%, Saving model...')
                torch.save(net.state_dict(), os.path.join(MODEL_PATH, "model.pkl"))
                best = accuracy
            torch.save(net.state_dict(), os.path.join(MODEL_PATH, "model.pkl"))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()
