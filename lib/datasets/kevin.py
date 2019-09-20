# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
 
class Kevin(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=20,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=473, 
                 crop_size=(473, 473), 
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):

        super(Kevin, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        # self.label_mapping = {-1: ignore_label, 0: ignore_label, 
        #                       1: ignore_label, 2: ignore_label, 
        #                       3: ignore_label, 4: ignore_label, 
        #                       5: ignore_label, 6: ignore_label, 
        #                       7: 0, 8: 1, 9: ignore_label, 
        #                       10: ignore_label, 11: 2, 12: 3, 
        #                       13: 4, 14: ignore_label, 15: ignore_label, 
        #                       16: ignore_label, 17: 5, 18: ignore_label, 
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        #                       25: 12, 26: 13, 27: 14, 28: 15, 
        #                       29: ignore_label, 30: ignore_label, 
        #                       31: 16, 32: 17, 33: 18}
                              
        self.label_mapping = {-1: -1, 0: 0, 
                              1: 1, 2: 2, 
                              3: 3, 4: 4, 
                              5: 5, 6: 6, 
                              7: 7, 8: 8, 9: 9, 
                              10: 10, 11: 11, 12: 12, 
                              13: 13, 14: 14, 15: 15, 
                              16: 16, 17: 17, 18: 18, 
                              19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
                              25: 25, 26: 26, 27: 27, 28: 28, 
                              29: 29, 30: 30, 
                              31: 31, 32: 32, 33: 33}

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
    
    def read_files(self):
        files = []
        for item in self.img_list:
            if 'train' in self.list_path:
                image_path, label_path, label_rev_path, _ = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "label_rev": label_rev_path, 
                          "name": name,}
            elif 'val' in self.list_path:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {"img": image_path,
                          "label": label_path,
                          "name": name,}
            elif 'infer' in self.list_path:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                sample = {"img": image_path[0],
                          "name": name,} # weird
            else:
                raise NotImplementedError('Unknown subset.')
            files.append(sample)
        return files

    def resize_image(self, image, size): 
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
        return image

    # def resize_image(self, image, label, size): 
    #     image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR) 
    #     label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    #     return image, label
     
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(
                    self.root, 'kevin/', item["img"]), 
                    cv2.IMREAD_COLOR)
        # label = cv2.imread(os.path.join(
        #             self.root, 'kevin/', item["label"]),
        #             cv2.IMREAD_GRAYSCALE)
        size = image.shape

        if 'testval' in self.list_path:
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name
            # return image.copy(), label.copy(), np.array(size), name

        if 'infer' in self.list_path:
            image = cv2.resize(image, self.crop_size, 
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        # if self.flip:
        #     flip = np.random.choice(2) * 2 - 1
        #     image = image[:, ::flip, :] 
        #     if flip == -1:
        #         label = cv2.imread(os.path.join(self.root, 
        #                     'kevin/', item["label_rev"]),
        #                     cv2.IMREAD_GRAYSCALE)
        #
        # image, label = self.resize_image(image, label, self.crop_size)
        # image, label = self.gen_sample(image, label, self.multi_scale, False)
        #
        # return image.copy(), label.copy(), np.array(size), name
        
        image = self.resize_image(image, self.crop_size)
        image = self.gen_sample(image, self.multi_scale, False)

        return image.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.upsample(input=pred, 
                          size=(size[-2], size[-1]), 
                          mode='bilinear')        
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output, 
                            size=(size[-2], size[-1]), 
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred[:,14,:,:] = flip_output[:,15,:,:]
            flip_pred[:,15,:,:] = flip_output[:,14,:,:]
            flip_pred[:,16,:,:] = flip_output[:,17,:,:]
            flip_pred[:,17,:,:] = flip_output[:,16,:,:]
            flip_pred[:,18,:,:] = flip_output[:,19,:,:]
            flip_pred[:,19,:,:] = flip_output[:,18,:,:]
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name="name"):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name+'.png'))

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    
