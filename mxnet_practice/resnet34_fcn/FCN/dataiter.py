import mxnet as mx
import numpy as np
import sys, os
from  PIL import Image
import os.path as osp
from mxnet.io import DataBatch, DataIter

class CustomIter(DataIter):
    """FileIter object for image semantic segmentation.
    Parameters
    ----------

    dataset : string
        dataset
    split : string
        data split
        the list file of images and labels, whose each line is in the format:
        image_path \t label_path
    data_root : string
        the root data directory
    data_name : string
        the data name used in the network input
    label_name : string
        the label name used in SoftmaxOutput
    sampler: str
        how to shuffle the samples per epoch
    has_gt: bool
        if there are ground truth labels
    batch_images : int
        the number of images per batch
    meta : dict
        dataset specifications

    prefetch_threads: int
        the number of prefetchers
    prefetcher_type: string
        the type of prefechers, e.g., process/thread
    """
    def __init__(self,
                 data_lst,
                 data_root,
                 dataset,
                 data_name = 'data',
                 label_name = 'label',
                 data_shape = [1, 3, 320, 320],
                 label_shape = [1, 1, 320/8, 320/8],
                 sampler = 'random',
                 batch_size = 4,
                 label_stride = 8,
                 crop_h = 320,
                 crop_w = 320,
                 use_flip = True,
                 resize_range = [0.9, 1.1]
                 ):

        super(CustomIter, self).__init__()

        self._dataset = dataset
        self._data_root = data_root
        self._label_stride = label_stride
        self._sampler = sampler

        self._crop_h = crop_h
        self._crop_w = crop_w

        self._data_lst = data_lst

        with open(self._data_lst) as f:
            self._name_list = f.readlines()

        self._perm_len = len(self._name_list)
        self._name_id = np.arange(self._perm_len)

        self._batch_size = batch_size
        self._cur_pointer = 0

        self._use_flip = use_flip
        self._resize_range = resize_range

        self._data_name = data_name
        self._label_name = label_name
        self._data_shape = data_shape
        self._label_shape = label_shape

        self._provide_data = [(d_name, d_shape) for d_name, d_shape in zip(self._data_name, self._data_shape)]
        self._provide_label = [(l_name, l_shape) for l_name, l_shape in zip(self._label_name, self._label_shape)]

    @property
    def batch_images(self):
        return self._batch_size

    @property
    def batches_per_epoch(self):
        return self._perm_len // self._batch_size

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        #the data provided for 4 gpu, the whole batch size is 16 while every mini batch has 4 images
        return self._provide_data

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return self._provide_label

    def parse_lst(self, path):
        if self._dataset == 'pascal':
            idxs = path.strip()
            abs_img_path = self._data_root+'/JPEGImages/'+idxs+'.jpg'
            abs_lab_path = self._data_root+'/SegmentationClass/'+idxs+'.png'
        else:
            raise NotImplementedError
        self._cur_image = abs_img_path
        self._cur_label = abs_lab_path
        return abs_img_path, abs_lab_path

    def random_crop(self, image, label):
        _h, _w, _ = image.shape
        _real_h, _real_w = int(max(_h, self._crop_h)), int(max(_w, self._crop_w))
        paded_image = np.zeros((_real_h, _real_w, 3), np.uint8)
        paded_label = np.zeros((_real_h, _real_w), np.uint8)
        paded_image[:_h, :_w, :] = image
        paded_label[:_h, :_w] = label

        _scale1, _scale2 = np.random.uniform(self._resize_range[0], self._resize_range[1], size=(2,))
        real_crop_h, real_crop_w = int(min(self._crop_h*_scale1, _real_h)), int(min(self._crop_w*_scale2, _real_w))

        if real_crop_h == _real_h:
            _sh = 0
        else:
            _sh = np.random.randint(low=0, high=_real_h-real_crop_h, size=())
        if real_crop_w == _real_w:
            _sw = 0
        else:
            _sw = np.random.randint(low=0, high=_real_w-real_crop_w, size=())

        tmp_crop_image = paded_image[_sh:_sh+real_crop_h, _sw:_sw+real_crop_w, :]
        tmp_crop_label = paded_label[_sh:_sh+real_crop_h, _sw:_sw+real_crop_w]

        rsz_image = np.array(Image.fromarray(tmp_crop_image).resize((self._crop_w, self._crop_h), resample=Image.BILINEAR))
        rsz_label = np.array(Image.fromarray(tmp_crop_label).resize((self._crop_w, self._crop_h), resample=Image.NEAREST))

        return rsz_image.transpose(2, 0, 1), rsz_label[self._label_stride//2::self._label_stride, self._label_stride//2::self._label_stride]

    def reset(self):
        if self._sampler == 'random':
            np.random.shuffle(self._name_id)
        self._name_id = self._name_id[:(self._perm_len//self._batch_size)*self._batch_size]
        print self._name_id
        self._cur_pointer = 0

    def next(self):
        if self._cur_pointer+self._batch_size < len(self._name_id):
            cur_images = np.zeros((self._batch_size, 3, self._crop_h, self._crop_w))
            cur_labels = np.zeros((self._batch_size, 1, self._crop_h/self._label_stride, self._crop_w/self._label_stride))
            for i in range(self._batch_size):
                image_path, label_path = self.parse_lst(self._name_list[self._name_id[i+self._cur_pointer]])

                ori_image = np.array(Image.open(image_path))
                ori_label = np.array(Image.open(label_path))
                if self._use_flip:
                    do_flip = np.random.randint(2) == 0
                    if do_flip:
                        ori_image = ori_image[:, ::-1, :]
                        ori_label = ori_label[:, ::-1]

                crop_image, crop_label = self.random_crop(image=ori_image, label=ori_label)
                cur_images[i, ...] = crop_image
                cur_labels[i, ...] = crop_label
        else:
            raise StopIteration
        self._cur_pointer += self._batch_size
        return DataBatch(data=[mx.nd.array(cur_images)], label=[mx.nd.array(cur_labels)])
