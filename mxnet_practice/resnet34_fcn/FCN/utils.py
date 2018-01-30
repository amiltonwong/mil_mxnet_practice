import numpy as np
from mxnet.metric import EvalMetric
import mxnet as mx
import logging

class FixedScheduler(mx.lr_scheduler.LRScheduler):
    def __call__(self, num_update):
        return self.base_lr

class LinearScheduler(mx.lr_scheduler.LRScheduler):
    """Reduce learning rate linearly
    Assume the weight has been updated by n times, then the learning rate will
    be
    base_lr * (1 - n/iters)
    Parameters
    ----------
    step: int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, updates, frequency=0, stop_lr=-1., offset=0):
        super(LinearScheduler, self).__init__()
        if updates < 1:
            raise ValueError('Schedule required max number of updates to be greater than 1 round')
        self._updates = updates
        self._frequency = frequency
        self._stop_lr = stop_lr
        self._offset = offset
        self._pre_updates = -1

    def __call__(self, num_update):
        """
        Call to schedule current learning rate
        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """
        now_update = self._offset + num_update
        if now_update > self._updates:
            if self._pre_updates != num_update:
                print 'Exceeds the number of updates, {} > {}'.format(now_update, self._updates)
                self._pre_updates = num_update
            now_update = self._updates

        lr = self.base_lr * (1 - float(now_update) / self._updates)

        if self._stop_lr > 0. and lr < self._stop_lr:
            lr = self._stop_lr
        if self._frequency > 0 and num_update % self._frequency == 0 and self._pre_updates != num_update:
            logging.info('Update[%d]: Current learning rate is %0.5e', num_update, lr)
            self._pre_updates = num_update
        return lr

class Custom_Accuracy(EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    """
    def __init__(self, axis=1, name='fcn-acc',
                 output_names=None, label_names=None):
        super(Custom_Accuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        gt_label = labels[0]
        pred_label = preds[0]

        pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)

        pred_label = pred_label.asnumpy()
        gt_label = gt_label.asnumpy()
        gt_label = gt_label.ravel()
        pred_label = pred_label.ravel()

        valid_flag = gt_label != 255
        gt_label = gt_label[valid_flag]
        pred_label =pred_label[valid_flag]

        self.sum_metric = float((gt_label==pred_label).sum())
        self.num_inst = float(valid_flag.sum()+(valid_flag.sum()==0))


        return (self.sum_metric, self.num_inst)

