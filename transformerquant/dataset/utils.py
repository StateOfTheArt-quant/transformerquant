#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from functools import reduce


def split(np_data, window=5, step=1, offset=0, keep_tail=True):
    """
    :param np_data: numpy data
    :param window: int, size of window default=5
    :param step: int, size between two windows default=1
    :param offset: int, first window offset default=0
    :param keep_tail: Boolean , {True : save tail of data,; False : possible not save tail of data} default True
    :return: list within numpy data

    Examples::
        >>> data = np.array([1,2,3,4,5,6,7,8,9,10])
        >>> # keep_tail is True
        >>> split_list = split(data, window=4, step=5, offset=0, keep_tail=True)
        >>> split_list  # [array([1]), array([2, 3, 4, 5]), array([ 7,  8,  9, 10])]
        >>> # keep_tail is False
        >>> split_list = split(data, window=4, step=5, offset=0, keep_tail=False)
        >>> split_list # [array([1, 2, 3, 4]), array([6, 7, 8, 9]), array([10])]
    """
    window, step, offset = int(window), int(step), int(offset)
    sample_list = []
    index = int((len(np_data) - window - offset) / step) + 1 #total steps
    remain = int(len(np_data) - window - offset - (index - 1) * step)
    #print('remain : ', remain)
    
    if keep_tail:
        start_index = remain+offset#
        if remain > 0:
            sample_list.append(np_data[offset:offset+remain])
        for i in range(index):
            window_data = np_data[start_index + i * step : start_index + window + i * step]
            sample_list.append(window_data)
    else:
        start_index = offset
        for i in range(index):
            window_data = np_data[start_index + i * step : start_index + window + i * step]
            sample_list.append(window_data)
        if remain > 0:
            sample_list.append(np_data[-remain:])

    return sample_list


def split_sample(np_data, window=5, step=1, offset=0, keep_tail=True, merge_remain=False):
    """
    :param np_data: numpy data
    :param window: int, size of window default=5
    :param step: int, size between two windows default=1
    :param offset: int, first window offset default=0
    :param keep_tail: Boolean , {True : save tail of data,; False : possible not save tail of data} default True
    :param merge_remain: Boolean , {True: and if keep_tail is True, the first sample include remain sample, 
                                           elif keep_tail is Flase, the last sample include remain sample.
                                 Flase: the sample decide by value of keep_tail
                                }
    :return: list within numpy data

    Examples::
        
        >>> # use to split data set
        >>> import numpy as np
        >>> data = np.array(range(1, 11))
        >>> window_train = 5
        >>> window_test = 3
        >>> # keep_tail=False, merge_remain=False 
        >>> train_data = split_sample(data, window=window_train, step=window_test, offset=0, keep_tail=False, merge_remain=False)
        >>> train_data
        [array([1, 2, 3, 4, 5]), array([4, 5, 6, 7, 8])]
        >>> test_data = split_sample(data, window=window_test, step=window_test, offset=window_train, keep_tail=False, merge_remain=True)
        [array([ 6,  7,  8,  9, 10])]
        
        >>> # use to split sample
        >>> data = np.array(range(30)).reshape(6, 5)
        >>> # keep_tail=True, merge_remain=False
        >>> sample1 = split_sample(data, window=3, step=2, offset=0, keep_tail=True, merge_remain=False)
        >>> sample1
        [array([[ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19]]),
         array([[15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24],
                [25, 26, 27, 28, 29]])]
        
        >>> # keep_tail=False, merge_remain=False
        >>> sample2 = split_sample(data, window=3, step=2, offset=0, keep_tail=False, merge_remain=False)
        >>> sample2
        [array([[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [10, 11, 12, 13, 14]]),
         array([[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24]])]
    """
    index = int((len(np_data) - window - offset) / step) + 1
    remain = len(np_data) - window - offset - (index - 1) * step
    sample_list = split(np_data, window=window, step=step, offset=offset, keep_tail=keep_tail)
    if remain:
        if keep_tail:
            idx = 1
        else:
            idx = -1
        
        if not merge_remain:
            return sample_list[idx:] if idx==1 else sample_list[:idx]
        else:
            sample_list[idx-1] = np.concatenate([sample_list[idx-1], sample_list[idx]])
            del sample_list[idx]
            return sample_list 
    else:
        return sample_list 

def split_train_test(np_data, train_ratio=0.7):
    """
    :param np_data: numpy data
    :param train_ratio: float, {0~1} The percentage of train set. example 0.7
    :return: numpy {train set, test set}
    """
    train_data = np_data[:int(len(np_data)*train_ratio)]
    test_data = np_data[int(len(np_data)*train_ratio):]
    return train_data, test_data

def split_train_val_test(np_data, val_ratio=0.2, test_ratio=0.1):
    """
    :param np_data: numpy data
    :param val_ratio: float, {0~1} The percentage of validation set. example 0.2
    :param test_ratio:  float, {0~1} The percentage of test set. example 0.1
    :return: numpy {train set, validation set, test set}
    """
    train_data, val_test_data = split_train_test(np_data, train_ratio=1-val_ratio-test_ratio)
    if 1-val_ratio-test_ratio == 1:
        return train_data, val_test_data[0:],val_test_data[-1:]
    else:
        val_data, test_data = split_train_test(val_test_data, train_ratio=val_ratio/(val_ratio+test_ratio))
        return train_data, val_data, test_data