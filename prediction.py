# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
from path import MODEL_PATH, DATA_PATH, DATA_ID
import crnn
import utils
from character import cut_character

model_path = os.path.join(MODEL_PATH, 'model.pkl')
alphabet = "abcdefghijklmnopqrstuvwxyz-' "
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Prediction(FlyAI):

    def load_model(self):
        """
        模型初始化，必须在此方法中加载模型
        """
        self.model = crnn.CRNN(imgH=32, nc=1, nclass=len(alphabet) + 1, nh=256, leakyRelu=True)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        # self.model = torch.load(model_path)

    def predict(self, image_path):
        """
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":"image\/172691.jpg"}
        :return: 模型预测成功之后返回给系统样例 {"label":"ZASSEOR"}
        """
        # net = self.model
        self.model.eval()
        with torch.no_grad():
            try:
                # img = cv2.imread(image_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cut_character(image_path)
                img = Image.fromarray(np.uint8(img))
            except:
                return {"label": 'EMPTY'}

            imgH = 32
            imgW = 100
            w, h = img.size
            max_ratio = w / float(h)

            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH, imgW)  # assure imgH >= imgW
            img = img.resize((imgW, imgH), Image.BILINEAR)
            toTensor = transforms.ToTensor()
            img = toTensor(img)
            img.sub_(0.5).div_(0.5)
            img = img.unsqueeze(0)

            # print(img.dtype)

            preds = self.model(img.to(device))
            preds_size = torch.IntTensor([preds.size(0)])
            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

            converter = utils.strLabelConverter(alphabet)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            sim_preds = sim_preds.upper()
            # print(sim_preds)
            if len(sim_preds) < 2:
                sim_preds = 'EMPTY'

        return {"label": sim_preds}


if __name__ == '__main__':
    image_path = 'data/test/image/1191.jpg'

    predict = Prediction()
    predict.load_model()
    print(predict.predict(image_path))
