import mxnet as mx
import numpy as np
import os
from PIL import Image
import logging
import os.path as osp
from dataiter import CustomIter
from mxnet.metric import CompositeEvalMetric
from utils import Custom_Accuracy, FixedScheduler, LinearScheduler
import time
import argparse

base_lr = 0.0008
momentum = 0.9
wd = 0.0005
lr_type = 'fixed'

pretrain_weight_path = 'resnet-34-0000.params'
input_h = 320
input_w = 320
batch_size = 16
model_save_dir = '/data-sdb/shuo.cheng/FCN/finetune'
data_root = '/data-sdb/shuo.cheng/dataset/VOCdevkit/VOC2012'
data_list = '/data-sdb/shuo.cheng/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
from_epoch = 0
num_epochs = 100
phase = 'train'

def get_network():
    from networks import fcn34
    return fcn34()

def get_scheduler():
    if lr_type == 'fixed':
        return FixedScheduler()
    elif lr_type == 'linear':
        return LinearScheduler(updates=num_epochs+1, frequency=50,
                                            stop_lr=min(base_lr/100., 1e-6),
                                            offset=from_epoch)
    else:
        raise NotImplementedError

def get_optimizer():
    sgd = mx.optimizer.create('sgd',
                              rescale_grad=1.0,
                              learning_rate=base_lr,
                              momentum=momentum,
                              wd=wd,
                              lr_scheduler=get_scheduler()
                              )
    return sgd

def get_module():
    contexts = [mx.gpu(int(_)) for _ in [0,1]]
    net = get_network()
    data_shape = (batch_size, 3, input_h, input_w)
    label_shape = (batch_size, 1, input_h//8, input_w//8)

    mod = mx.mod.Module(symbol=net,
                        data_names=('data',),
                        label_names=('label',),
                        context=contexts
                        )

    mod.bind(data_shapes=[('data', data_shape)],
             label_shapes=[('label', label_shape)],
             for_training=True,
             force_rebind=True
             )
    return mod

def load_params_from_file(path):
    """Load model checkpoint from file."""
    save_dict = mx.nd.load(path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

def train():
    if pretrain_weight_path is not None:
        net_args, net_auxs = load_params_from_file(pretrain_weight_path)
    else:
        net_args, net_auxs = None, None
    to_model = osp.join(model_save_dir, '{}_ep'.format('FCN'))
    mod = get_module()
    opt = get_optimizer()

    dataiter = CustomIter(data_lst=data_list, dataset='pascal', data_root=data_root,
                          batch_size=batch_size, crop_h=input_h, crop_w=input_w, label_stride=8, sampler='random')
    custom_eval = CompositeEvalMetric()
    custom_eval.add(Custom_Accuracy())
    dataiter.reset()
    mod.fit(
        dataiter,
        eval_metric=custom_eval,
        batch_end_callback=mx.callback.Speedometer(batch_size, 1),
        epoch_end_callback=mx.callback.do_checkpoint(to_model),
        kvstore='local',
        begin_epoch=from_epoch,
        num_epoch=num_epochs,
        optimizer=opt,
        initializer=mx.init.Xavier(),
        arg_params=net_args,
        aux_params=net_auxs,
        allow_missing=True,
    )

def set_logger(output_dir=None, log_file=None, debug=False):
    head = '%(asctime)-15s Host %(message)s'
    logger_level = logging.INFO if not debug else logging.DEBUG
    if all((output_dir, log_file)) and len(log_file) > 0:
        logger = logging.getLogger()
        log_path = osp.join(output_dir, log_file)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_level)
    else:
        logging.basicConfig(level=logger_level, format=head)
        logger = logging.getLogger()
    return logger

def val():
    load_weight_path = '/data-sdb/shuo.cheng/FCN/finished/FCN_ep-0100.params'
    test_lst = '/data-sdb/shuo.cheng/dataset/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'
    save_res_dir = 'results'

    net_args, net_auxs = load_params_from_file(load_weight_path)
    net = get_network()
    mod = mx.mod.Module(symbol=net,
                        data_names=('data',),
                        label_names=('label',),
                        context=mx.gpu(0))

    if not osp.exists(save_res_dir):
        os.makedirs(save_res_dir)

    with open(test_lst) as f:
        test_paths = f.readlines()
        for cur_path in test_paths:

            img = np.array(Image.open(data_root+'/JPEGImages/'+cur_path.strip()+'.jpg'))
            _h, _w, _ = img.shape
            mod.bind(data_shapes=[('data', (1, 3, _h, _w))],
                     label_shapes=None,
                     force_rebind=True,
                     for_training=False,
                     )
            mod.init_params(arg_params=net_args, aux_params=net_auxs, allow_missing=False, force_init=False)
            data_sym = mx.nd.array(img.transpose(2, 0, 1).reshape((1, 3, _h, _w)))
            data_batch = mx.io.DataBatch(data=[data_sym], label=None)
            print '\nTesting image %s ...'%(cur_path.strip())
            tic = time.time()
            mod.forward(data_batch=data_batch, is_train=False)
            cur_predict = mod.get_outputs()[0].asnumpy()
            print 'Finished in %.3f s.\n'%(time.time()-tic)
            rsz_predict = resize_pred(pred=cur_predict[0], _h=_h, _w=_w)
            final_res = np.argmax(rsz_predict, axis=0).astype(np.uint8)
            Image.fromarray(final_res).save(save_res_dir+'/%s.png'%cur_path.strip())

def resize_pred(pred, _h, _w):
    c, _, _ = pred.shape
    res = np.zeros((c, _h, _w), np.float)
    for i in range(c):
        rsz = np.array(Image.fromarray(pred[i, ...]).resize((_w, _h), resample=Image.BICUBIC))
        res[i, ...] = rsz
    return res

if __name__  == '__main__':
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)
    logger = set_logger(model_save_dir, 'log.txt')
    if phase is 'train':
        train()
    elif phase is 'val':
        val()



