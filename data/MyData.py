"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot

undated by Wyn Mew
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

vm169_CLASSES = (  # always index 0
    'background', 'ssd0007', 'ssd0012', 'ssd0033', 'ssd0068', 'ssd0075',
    'ssd0082', 'ssd0202', 'ssd0281', 'ssd0282', 'ssd0283',
    'ssd0284', 'ssd0290', 'ssd0430', 'ssd0433', 'ssd0437',
    'ssd0438', 'ssd0471', 'ssd0475', 'ssd0486', 'ssd0489',
    'ssd0492', 'ssd0505', 'ssd0506', 'ssd0507', 'ssd0524',
    'ssd0526', 'ssd0533', 'ssd0555', 'ssd0557', 'ssd0597')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(vm169_CLASSES, range(len(vm169_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            '''
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            '''
            name = obj.find('name').text.lower().strip()
            #print(name)
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            label1 = int(label_idx / 6)
            label2 = label_idx % 6
            #print(label_idx)
            #print(label1)
            #print(label2)
            # label transform
            bndbox.append(label1) # 0 ~ 5
            bndbox.append(label2) # 0 ~ 5
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class vm169Detection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    #dataset = vm169Detection(vm169root, train_sets, SSDAugmentation(
    #    ssd_dim, means), AnnotationTransform())

    def __init__(self, root, image_sets, transform=None, target_transform=None,
                 dataset_name='vm169'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s','vm169', 'images', '%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'vm169', 'images','%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (name) in image_sets:
            rootpath = os.path.join(self.root)
            for line in open(os.path.join(name)):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_rid = self.ids[index]
        #print(img_rid)
        imgstr = img_rid[1].split('\t')
        img_id = []
        img_id.append(img_rid[0])
        img_id.append(imgstr[0])
        img_id.append(imgstr[1])

        target = ET.parse(self._annopath % (img_id[0], img_id[1], img_id[2])).getroot()
        img = cv2.imread(self._imgpath % (img_id[0], img_id[1], img_id[2]))
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, label1, label2 = self.transform(img, target[:, :4],target[:, 4], target[:, 5])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(label1, axis=1), np.expand_dims(label2, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_rid = self.ids[index]
        # print(img_rid)
        imgstr = img_rid[1].split('\t')
        img_id = []
        img_id.append(img_rid[0])
        img_id.append(imgstr[0])
        img_id.append(imgstr[1])
        return cv2.imread(self._imgpath % (img_id[0], img_id[1], img_id[2]), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_rid = self.ids[index]
        # print(img_rid)
        imgstr = img_rid[1].split('\t')
        img_id = []
        img_id.append(img_rid[0])
        img_id.append(imgstr[0])
        img_id.append(imgstr[1])
        anno = ET.parse(self._annopath % (img_id[0], img_id[1], img_id[2])).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

'''
from utils.augmentations import SSDAugmentation
train_sets = [('WynTrain30')]
ssd_dim = 300  # only support 300 now
means = (104, 117, 123)  # only support voc now
vm169root='/home/chenli/Documents/data'
dataset = vm169Detection(vm169root, train_sets, SSDAugmentation(
        ssd_dim, means), AnnotationTransform())
dataset.pull_item(1)
        
#testset = vm169Detection(args.vm169_root, [('WynTest30')], None, AnnotationTransform())
'''
