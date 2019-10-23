# -*- coding: utf-8 -*-
# Copyright StateOfTheArt.quant. 
#
# * Commercial Usage: please contact allen.across@gmail.com
# * Non-Commercial Usage:
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os
import logging
import numpy as np
import pandas as pd
from functools import reduce
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import torch
from torch.utils.data import TensorDataset
import pdb
import re

#
from transformerquant.dataset.utils import split_sample, split_train_val_test


logger = logging.getLogger(__name__)

def list_flatten(a_list):
    return [y for l in a_list for y in list_flatten(l)] if isinstance(a_list, list) else [a_list]

class SamplerBase(object):

    @staticmethod
    def create(config):
        kwargs = vars(config)
        sequence_window = kwargs.pop('sequence_window', 1)
        frequency_x = kwargs.pop('frequency_x', '1d')
        interval_depart = kwargs.pop('interval_depart', True)
        return SamplerBase(sequence_window, frequency_x, interval_depart)

    def __init__(self, sequence_window=1, frequency_x='1d', interval_depart=True):
        self.sequence_window = sequence_window
        self.frequency_x = frequency_x
        self.interval_depart = interval_depart
    
    
    def generate_sample(self, featured_df):
        assert isinstance(self.sequence_window, int), "sequence window must be int"
        if featured_df.empty:
            return [np.array([]), np.array([]), []]

        if self.sequence_window == 1:
            all_x_sample_np, all_y_sample_np, all_datetime_sample_list = self._split_to_general_sample(featured_df)
        elif self.sequence_window > 1 and (self.frequency_x[-1] == 'd' or self.interval_depart is False):
            all_x_sample_np, all_y_sample_np, all_datetime_sample_list = self._split_to_seq_sample(featured_df)
        elif self.sequence_window > 1 and (self.frequency_x[-1] == 'm' or self.interval_depart is True):
            all_x_sample_np, all_y_sample_np, all_datetime_sample_list = self._split_minute_to_seq_sample(featured_df)
        else:
            raise KeyError("sequence_widnow and frequency_x is not available")
        return all_x_sample_np, all_y_sample_np, all_datetime_sample_list

    # sequence_window = 1的日，分钟数据
    def _split_to_general_sample(self, ndays_df):
        ndays_feature_df = ndays_df.iloc[:,:-1]
        ndays_label_df = ndays_df.iloc[:,-1]
        return ndays_feature_df.values, ndays_label_df.values, ndays_df.index.tolist()

    # split sequence_window > 1的日，分钟数据,分钟数据日连续
    def _split_to_seq_sample(self, ndays_df):
        ndays_feature_df = ndays_df.iloc[:,:-1]
        ndays_label_df = ndays_df.iloc[:, -1]

        x_sample_list = split_sample(ndays_feature_df.values, window=self.sequence_window, step=1, offset=0, keep_tail=True)
        y_sample_list = split_sample(ndays_label_df.values, window=self.sequence_window, step=1, offset=0, keep_tail=True)
        datetime_list = split_sample(ndays_df.index, window=self.sequence_window, step=1, offset=0, keep_tail=True)
        datetime_sample_list = [datetime[-1] for datetime in datetime_list]
        y_sample_list = [y_sample[-1] for y_sample in y_sample_list]
        return np.array(x_sample_list), np.array(y_sample_list), datetime_sample_list

    # split sequence_window > 1的分钟数据, 分钟数据日间隔开, 日内隔开
    def _split_minute_to_seq_sample(self, n_days_minute_df):
        # 按天进行groupby, return a dict with key is date and value is a DatetimeIndex conatiains all datetimes in that date
        date_datetime_dict = n_days_minute_df.groupby(n_days_minute_df.index.date).groups.items()

        def convert_df_to_sample(date, idx, ndays_df):
            oneday_feature_df = ndays_df.ix[idx]
            # A dict that containe one day sample with 'x_sample','y_sample','dateime_list' keys
            sample_container = self._split_intraday_sequence_sample(date, oneday_feature_df)
            return sample_container
        # 返回值是sample_container中的values是list而不是numpy，不再使用reduce
        sample_container_list = list(map(lambda x: convert_df_to_sample(x[0], x[1], n_days_minute_df), date_datetime_dict))
        #
        all_x_sample_list = list(map(lambda x: x['x_sample'], sample_container_list))
        all_y_sample_list = list(map(lambda x: x['y_sample'], sample_container_list))
        all_datetime_sample_list = list(map(lambda x: x['datetime_list'], sample_container_list))
        all_x_sample_np = np.array(list_flatten(all_x_sample_list))  # 如果传入的数据条数不够一个样本长度，这里将是[]
        all_y_sample_np = np.array(list_flatten(all_y_sample_list)) # 如果传入的数据条数不够一个样本长度，这里将是[]
        return all_x_sample_np, all_y_sample_np, list_flatten(all_datetime_sample_list)

    def _split_intraday_sequence_sample(self, date, oneday_feature_df):
        sample_container = {}
        split_datetime = pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=12, minute=0, second=0)

        x_sample_up_list, y_sample_up_list, datetime_sample_up_list = \
            self._split_intraday_sample_base(oneday_feature_df.truncate(after=split_datetime)) # 对上午进行切分
        x_sample_down_list, y_sample_down_list, datetime_sample_down_list = \
            self._split_intraday_sample_base(oneday_feature_df.truncate(before=split_datetime)) # 对下午进行切分

        x_sample_list = list_flatten([x_sample_up_list, x_sample_down_list])
        y_sample_list = list_flatten([y_sample_up_list, y_sample_down_list])
        datetime_sample_list = list_flatten([datetime_sample_up_list, datetime_sample_down_list])

        # offer list type
        sample_container['x_sample'] = x_sample_list
        sample_container['y_sample'] = y_sample_list
        sample_container['datetime_list'] = datetime_sample_list
        return sample_container

    def _split_intraday_sample_base(self, halt_intraday_feature_df):
        if len(halt_intraday_feature_df) == 0:
            return [],[],[]
        hatday_feature_df = halt_intraday_feature_df.iloc[:,:-1]
        hatday_label_series = halt_intraday_feature_df.iloc[:,-1]
        x_sample_list = split_sample(hatday_feature_df.values, window=self.sequence_window, step=1, offset=0, keep_tail=True)
        y_sample_list = split_sample(hatday_label_series.values, window=self.sequence_window, step=1, offset=0, keep_tail=True)
        datetime_list = split_sample(halt_intraday_feature_df.index, window=self.sequence_window, step=1, offset=0, keep_tail=True)
        datetime_sample_list = [datetime[-1] for datetime in datetime_list]
        y_sample_list = [y_sample[-1] for y_sample in y_sample_list]
        return x_sample_list, y_sample_list, datetime_sample_list


class Preprocesor(object):
    def __init__(self, order_book_id, frequency, process_nan=True):
        self.process_nan = process_nan
        self.save_name = order_book_id + '_' + frequency + '_MinMaxScaler.m'

    def fillna(self, featured_df):
        if self.process_nan is False:  # 不对nan处理
            return featured_df
        else:
            assert isinstance(featured_df, pd.DataFrame), "featured_df must be pandas DataFrame"
            cleaned_df = featured_df.fillna(method='ffill').fillna(method='bfill')
            return cleaned_df

    def fit_transform(self, featured_np, save_dir=None):
        assert isinstance(featured_np, np.ndarray), "featured_np must be numpy ndarray"
        self.scaler = MinMaxScaler()
        featured_scaled = self.scaler.fit_transform(featured_np)

        if save_dir is not None and os.path.exists(save_dir) is False:
            os.mkdir(save_dir)
        if save_dir is not None:
            joblib.dump(self.scaler, save_dir+self.save_name)
        return featured_scaled

    def transform(self, featured_np, save_dir=None):
        assert isinstance(featured_np, np.ndarray), "featured_np must be numpy ndarray"
        if save_dir is None:
            scaler = self.scaler
        else:
            if os.path.exists(save_dir) is False:
                raise KeyError('Error: save_path not exists')
            if os.path.isdir(save_dir):
                save_path = save_dir + self.save_name
            scaler = joblib.load(save_path)
        featured_scaled = scaler.transform(featured_np)
        return featured_scaled



class Sampler(SamplerBase):
    def __init__(self,
                 sequence_window=5,
                 frequency_x='5m',
                 interval_depart=True,
                 process_nan=True,
                 use_normalize = True,
                 saved_nomalizer_dir='/tmp/',
                 batch_size=8,
                 train_ratio=0.7,
                 val_ratio = 0.1,
                 test_ratio = 0.2,
                 task = 'classify'):
        super(Sampler, self).__init__(sequence_window, frequency_x, interval_depart)
        self.sequence_window = sequence_window
        self.use_normalize = use_normalize
        self.saved_nomalizer_dir = saved_nomalizer_dir
        self.task = task
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.process_nan = process_nan
        self.postfix_dict = {0:"train",1:"val",2:"test"}
        self.spliter = lambda x: split_train_val_test(x, val_ratio=val_ratio, test_ratio=test_ratio)


    def generate_sample(self, feature_container):
        container = {}
        symbol_sample_container_list = []
        logger.info("sampler begin...")
        for order_book_id, featured_df in feature_container.items():
            try:
                logger.info("start to process order_book_id :{}".format(order_book_id))
                preprocesor = Preprocesor(order_book_id, self.frequency_x, self.process_nan)
                cleaned_df = preprocesor.fillna(featured_df)

                # split train val test
                train_cleaned_df, val_cleaned_df, test_cleaned_df = self.spliter(cleaned_df)

                if self.use_normalize and self.train_ratio:  #
                    assert not train_cleaned_df.empty, "train_cleaned_df is empty"
                    assert not val_cleaned_df.empty, "val_cleaned_df is empty"
                    assert not test_cleaned_df.empty, "test_cleaned_df is empty"

                    train_normalized_feature_np = preprocesor.fit_transform(train_cleaned_df.iloc[:,:-1].values, self.saved_nomalizer_dir)
                    val_normalized_feature_np = preprocesor.transform(val_cleaned_df.iloc[:,:-1].values)
                    test_normalized_feature_np = preprocesor.transform(test_cleaned_df.iloc[:,:-1].values)

                    train_normalized_np = np.concatenate([train_normalized_feature_np, train_cleaned_df.iloc[:,-1].values.reshape(-1,1)], axis=1)
                    val_normalized_np = np.concatenate([val_normalized_feature_np, val_cleaned_df.iloc[:,-1].values.reshape(-1,1)], axis=1)
                    test_normalized_np = np.concatenate([test_normalized_feature_np, test_cleaned_df.iloc[:,-1].values.reshape(-1,1)], axis=1)

                    train_normalized_df = pd.DataFrame(train_normalized_np, index=train_cleaned_df.index)
                    val_normalized_df = pd.DataFrame(val_normalized_np, index=val_cleaned_df.index)
                    test_normalized_df = pd.DataFrame(test_normalized_np, index=test_cleaned_df.index)
                    normalized_df_list = [train_normalized_df, val_normalized_df, test_normalized_df]
                elif self.use_normalize and self.test_ratio == 1.0:
                    # pdb.set_trace()
                    test_normalized_feature_np= preprocesor.transform(test_cleaned_df.iloc[:,:-1].values, self.saved_nomalizer_dir)
                    test_normalized_np = np.concatenate([test_normalized_feature_np, test_cleaned_df.iloc[:, -1].values.reshape(-1, 1)], axis=1)
                    normalized_df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(test_normalized_np, index=test_cleaned_df.index)]
                else:
                    normalized_df_list = [train_cleaned_df, val_cleaned_df, test_cleaned_df]

                sample_list = list(map(lambda x: super(Sampler, self).generate_sample(x), normalized_df_list))

                symbol_sample_container = {}
                for index,sample in enumerate(sample_list):
                    symbol_sample_container['x_sample_'+ self.postfix_dict[index]] = sample[0]
                    symbol_sample_container['y_sample_'+ self.postfix_dict[index]] = sample[1]#.tolist()
                    symbol_sample_container['datetime_sample_'+ self.postfix_dict[index]] = sample[2]

                container[order_book_id] = symbol_sample_container
                symbol_sample_container_list.append(symbol_sample_container)

            except Exception as e:
                logger.debug('{} data process faild, error:{}'.format(order_book_id, e))

        x_sample_train_list = list(map(lambda x: x['x_sample_train'], symbol_sample_container_list))
        x_sample_val_list = list(map(lambda x: x['x_sample_val'], symbol_sample_container_list))
        x_sample_test_list = list(map(lambda x: x['x_sample_test'], symbol_sample_container_list))
        y_sample_train_list = list(map(lambda x: x['y_sample_train'], symbol_sample_container_list))
        y_sample_val_list = list(map(lambda x: x['y_sample_val'], symbol_sample_container_list))
        y_sample_test_list = list(map(lambda x: x['y_sample_test'], symbol_sample_container_list))
        #pdb.set_trace()
        x_sample_train_np = reduce(lambda x,y:np.concatenate([x,y]), x_sample_train_list).astype('float32')
        x_sample_val_np = reduce(lambda x,y:np.concatenate([x,y]), x_sample_val_list).astype('float32')
        x_sample_test_np = reduce(lambda x,y:np.concatenate([x,y]), x_sample_test_list).astype('float32')
        
        if self.task == 'classify':
            y_sample_train = reduce(lambda x,y:np.concatenate([x,y]), y_sample_train_list).reshape(-1).astype('int')
            y_sample_val = reduce(lambda x,y:np.concatenate([x,y]), y_sample_val_list).reshape(-1).astype('int')
            y_sample_test = reduce(lambda x,y:np.concatenate([x,y]), y_sample_test_list).reshape(-1).astype('int')
        else:
            y_sample_train = reduce(lambda x,y:np.concatenate([x,y]), y_sample_train_list).reshape(-1).astype('float32')
            y_sample_val = reduce(lambda x,y:np.concatenate([x,y]), y_sample_val_list).reshape(-1).astype('float32')
            y_sample_test = reduce(lambda x,y:np.concatenate([x,y]), y_sample_test_list).reshape(-1).astype('float32')

        dataset_train = TensorDataset(torch.tensor(x_sample_train_np), torch.tensor(y_sample_train))
        dataset_val = TensorDataset(torch.tensor(x_sample_val_np), torch.tensor(y_sample_val))
        dataset_test = TensorDataset(torch.tensor(x_sample_test_np), torch.tensor(y_sample_test))

        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)

        container['dataloader_train'] = self.dataloader_train
        container['dataloader_val'] = self.dataloader_val
        container['dataloader_test'] = self.dataloader_test
        logger.info("sampler successfully done")
        self.container = container
        return container

    def get_sample_by_dt(self, order_book_id, to_datetime, method=''):
        # '2018-11-12'
        # '2018-11-12 15:00:00'
        # '2018-11-12 00:00:00'
        # '2018-11-12 18:30:50'
        # 1. acquire the symbol in data container
        assert order_book_id in self.container.keys(), "symbol not in data container"
        sample_container = self.container[order_book_id]
        frequency_ = self.frequency_x[-1]

        if isinstance(to_datetime, str):
            re_time = re.findall(r'(\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2})', to_datetime)
            if len(re_time) == 0: # '2018-11-12'
                to_datetime = datetime.datetime.strptime(to_datetime, '%Y-%m-%d')
            else:
                to_datetime = datetime.datetime.strptime(to_datetime, '%Y-%m-%d %H:%M:%S')

        if frequency_[-1] == "d":
            hour_value = sample_container['datetime_sample_test'][0].hour  # "0"/"15"
            minute_value = 0
        else:
            hour_value = to_datetime.hour
            minute_value = to_datetime.minute
        # 转换后都不带秒 date_time
        time_stamp = pd.Timestamp(year=to_datetime.year, month=to_datetime.month, day=to_datetime.day, hour=hour_value,
                                  minute=minute_value, second=0)

        flag = time_stamp in sample_container['datetime_sample_test']
        if flag and method == '':
            index = sample_container['datetime_sample_test'].index(time_stamp)
        elif flag is False and method == '':
            print('cant find the sample in {}, but returning the sample in {}'
                  .format(time_stamp, sample_container['datetime_sample_test'][-1]))
            return None

        if method == "error":
            if frequency_ == 'd':  # 日数据
                error = list(map(lambda x: (time_stamp.date() - x.date()).days, sample_container['datetime_sample_test']))
            else:  # 分钟
                error = list(map(lambda x: (time_stamp - x).total_seconds(), sample_container['datetime_sample_test']))

            select = [x for x in error if x >= 0]
            if len(select) == 0: # 数据集的日期都在查询日期的后面
                print('first datetime in test is {},but you search is {}'
                      .format((sample_container['datetime_sample_test'][0]),(to_datetime)))
                return None
            else:
                index = error.index(np.min(select))
                print('return latest datetime:{}'.format(sample_container['datetime_sample_test'][index]))

        # 2. get the specific dateset in self.dataset_test according to index
        last_sample_np = sample_container['x_sample_test'][index]
        if self.sequence_window > 1:
            last_sample_np = last_sample_np.reshape(1, last_sample_np.shape[0], last_sample_np.shape[1])
        else:
            last_sample_np = last_sample_np.reshape(1, last_sample_np.shape[0])
        return torch.tensor(last_sample_np, dtype=torch.float32)

    def get_lastsample_datetime(self, order_book_id):
        assert order_book_id in self.container.keys(), "symbol not in data container"
        sample_container = self.container[order_book_id]
        return sample_container['datetime_sample_test'][-1]

