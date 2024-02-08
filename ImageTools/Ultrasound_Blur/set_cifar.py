# coding : utf-8
import os
import cv2
import torch
import json
import math
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt
import torch.nn as nn


device = torch.device(3)


class NewBlur:
    div = 20
    s = 160
    lr = 1e-2
    lim = 0.2

    def __init__(self, theta, x, y):
        self.theta = torch.Tensor([theta]).cuda(device).requires_grad_()
        self.x = torch.Tensor([x]).cuda(device).requires_grad_()
        self.y = torch.Tensor([y]).cuda(device).requires_grad_()
        self.grid = torch.linspace(0, 1, self.div).cuda(device)
        self.gx = 0
        self.gy = 0
        self.gtheta = 0

    def SquareBlur(self, img):
        if img.ndim == 3:
            img = img.unsqueeze(dim=0)
        img = torch.cat([img for _ in range(self.div)], 0)
        t1 = torch.cos(self.theta * self.grid)
        t2 = torch.sin(self.theta * self.grid)
        dx = self.x * self.grid
        dy = self.y * self.grid
        affineTensor1 = torch.stack([t1, t2, dx])
        affineTensor2 = torch.stack([-t2, t1, dy])
        affineTensor = torch.stack(
            [affineTensor1, affineTensor2]).permute(2, 0, 1)
        grid = F.affine_grid(affineTensor, img.shape, align_corners=False)
        imgs = F.grid_sample(
            img, grid, padding_mode="border", align_corners=False)
        img = imgs.mean(dim=0, keepdim=True)
        return img

    # def FGSM(self, targeted=False):
    #     if not targeted:
        # self.gx = 0.5 * self.gx + self.x.grad
        # self.gy = 0.5 * self.gy + self.y.grad
        # self.gtheta = 0.5 * self.gy + self.theta.grad
        # self.x.data += self.lr * self.gx
        # self.y.data += self.lr * self.gy
        # self.theta.data += self.lr * self.gtheta
    #     else:
    #         self.x.data -= self.lr * self.x.grad
    #         self.y.data -= self.lr * self.y.grad
    #         self.theta.data -= self.lr * self.theta.grad
    #     self.x.data.clamp_(min=-self.lim, max=self.lim)
    #     self.y.data.clamp_(min=-self.lim, max=self.lim)
    #     self.theta.data.clamp_(min=-self.lim, max=self.lim)
    #     self.x.grad.zero_()
    #     self.y.grad.zero_()
    #     self.theta.grad.zero_()

    def FGSM(self, targeted=False):
        if not targeted:
            self.x.data += self.lr * self.x.grad
            self.y.data += self.lr * self.y.grad
            self.theta.data += self.lr * self.theta.grad
        else:
            self.gx = 0.5 * self.gx + self.x.grad
            self.gy = 0.5 * self.gy + self.y.grad
            self.gtheta = 0.5 * self.gy + self.theta.grad
            self.x.data -= self.lr * self.gx
            self.y.data -= self.lr * self.gy
            self.theta.data -= self.lr * self.gtheta
        self.x.data.clamp_(min=-self.lim, max=self.lim)
        self.y.data.clamp_(min=-self.lim, max=self.lim)
        self.theta.data.clamp_(min=-self.lim, max=self.lim)
        self.x.grad.zero_()
        self.y.grad.zero_()
        self.theta.grad.zero_()


class OneImage:
    simpletf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    preprocess = transforms.Normalize(
        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    )
    toPIL = transforms.ToPILImage()

    def __init__(self, model=None):
        self.imgTensor = None
        self.blurImg = None
        self.model = model
        self.idx = None
        self.lastidx = None
        self.target = None
        self.targeted = False

    def setTarget(self, target):
        if isinstance(target, str):
            try:
                target = idx2cls.index(target)
            except:
                pass
        if isinstance(target, int):
            self.target = torch.tensor([target], dtype=int).cuda(device)
            self.targeted = True

    def setNewImage(self, imgPath):
        self.imgTensor = self.simpleLoadImg(imgPath).cuda(device).unsqueeze(0)
        self.getOriginalIndex()

    def setDefaultModel(self, model):
        self.model = model

    def simpleLoadImg(self, imgPath):
        img = Image.open(imgPath)
        return self.simpletf(img)

    def getOriginalIndex(self):
        with torch.no_grad():
            # print(self.imgTensor.shape)
            res = self.model(self.preprocess(self.imgTensor))
            self.idx = torch.max(res, dim=1)[1]
            print(torch.max(res, dim=1))

    def forward(self):
        res = self.model(self.preprocess(self.blurImg))
        self.lastidx = torch.max(res, dim=1)[1]
        print(res)
        # print(self.lastidx)
        # print(idx2cls[self.lastidx])
        if not self.targeted:
            loss = F.cross_entropy(res, self.idx)
        else:
            loss = F.cross_entropy(res, self.target)
        loss.backward()

    def saveOutImage(self, imgPath):
        self.toPIL(self.blurImg.squeeze(dim=0)).save(imgPath)


def getImageNetName(path="dataset/cifar.json"):
    cls_json = json.load(open(path))
    return [cls_json[str(k)][1] for k in range(len(cls_json))]


def attack(blurModule, oneImageModule):
    oneImageModule.blurImg = blurModule.SquareBlur(oneImageModule.imgTensor)
    # print(oneImageModule.blurImg.shape)
    # img = (
    #     torch.abs(oneImageModule.blurImg)
    #     .squeeze(0)
    #     .permute([1, 2, 0])
    #     .data.cpu()
    #     .numpy()
    # )
    # print(img.shape)
    # plt.imsave("img.png", img, format="png")
    oneImageModule.forward()


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block1 = ConvBlock(
            channels, channels, kernel_size, stride, padding)
        self.block2 = ConvBlock(
            channels, channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.block2(self.block1(x)) + x


cifar10_classifier = nn.Sequential(
    ConvBlock(in_channels=3, out_channels=64,
              kernel_size=3, stride=1, padding=1),
    ResBlock(channels=64),
    ResBlock(channels=64),
    ConvBlock(in_channels=64, out_channels=128,
              kernel_size=3, stride=2, padding=1),
    ConvBlock(in_channels=128, out_channels=128,
              kernel_size=3, stride=1, padding=1),
    ResBlock(channels=128),
    ConvBlock(in_channels=128, out_channels=256,
              kernel_size=3, stride=2, padding=1),
    ConvBlock(in_channels=256, out_channels=256,
              kernel_size=3, stride=1, padding=1),
    ResBlock(channels=256),
    ConvBlock(in_channels=256, out_channels=512,
              kernel_size=3, stride=2, padding=1),
    ConvBlock(in_channels=512, out_channels=512,
              kernel_size=3, stride=1, padding=1),
    ResBlock(channels=512),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(in_features=512, out_features=10),
).to(device)


def main():
    os.chdir("/home/usslab/shibo/graduate")
    model = cifar10_classifier
    model.load_state_dict(torch.load("cifar10.pt"))
    model.eval().cuda(device)

    cnn = OneImage(model)
    resu = np.zeros(10)
    pic_list = os.listdir("dataset/cifar/9")
    file = open("result/outputtarget9.txt", "w")

    for _ in pic_list:
        asr = 0
        # print(_, file=file)
        cnn.setNewImage("dataset/cifar/9/" + _)
        params = (0.0, 0.0, 0.0)
        print(_, file=file)
        #  target
        for idx in range(0, 10):
            print(f"{idx}:{idx2cls[idx]}", file=file)
            cnn.setTarget(idx)
            b = NewBlur(*params)
            flag = 0
            for i in range(2):
                attack(b, cnn)
                if cnn.lastidx == cnn.target:
                    flag = 1
                b.FGSM(targeted=cnn.targeted)
                # print(b.x.item(), b.y.item(), b.theta.item())
            if flag == 1:
                print(idx2cls[idx], "success!")
                resu[idx] = resu[idx] + 1
                print(b.x.item(), b.y.item(), b.theta.item(), file=file)
            asr += flag
            print(f"{asr}/{idx+1}", file=file)

        #   untarget
        # b = NewBlur(*params)
        # flag = 0
        # for i in range(2):
        #     attack(b, cnn)
        #     print(b.x.item(), b.y.item(), b.theta.item())
        #     if cnn.lastidx != cnn.idx:
        #         flag = 1
        #         break
        #     b.FGSM()
        # if flag == 1:
        #     print("success!")
        #     print(b.x.item(), b.y.item(), b.theta.item())
        #     resu[0] = resu[0] + 1

    print(resu)


idx2cls = getImageNetName()

if __name__ == "__main__":
    main()
    print()
