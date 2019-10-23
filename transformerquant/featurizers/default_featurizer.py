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

import featurizer.functors.talib as talib
from featurizer.functors.labeler import RegressionLabler
import featurizer.functors.volume_price as vp
import featurizer.functors.journalhub as jf
import featurizer.functors.time_series as tf

class DefaultFeaturizer(object):
    
    def __init__(self, fwd_returns_window=1, task="regression"):
        #
        if task == "regression":
            self.labeler = RegressionLabler(window=fwd_returns_window)
        
        self.pct_change = tf.PctChange(window=1) 
        #
        self.ROCP = talib.ROCP(timeperiod=1)
        self.MACD = talib.MACDRelated(fastperiod=12, slowperiod=26, signalperiod=9)
        self.RSI6 = talib.DemeanedRSI(timeperiod=6)
        self.RSI12 = talib.DemeanedRSI(timeperiod=12)
        self.RSI24 = talib.DemeanedRSI(timeperiod=24)
        
        self.RSIROCP6 = talib.RSIROCP(timeperiod=6)
        self.RSIROCP12 = talib.RSIROCP(timeperiod=12)
        self.RSIROCP24 = talib.RSIROCP(timeperiod=24)
        
        self.VROCP = talib.VolumeROCP()
        #BOLL
        self.BOLL = talib.BBANDS(timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        
        #MA
        self.MAROCP5 = talib.MAROCP(timeperiod=5)
        self.MAROCP10 = talib.MAROCP(timeperiod=10)
        self.MAROCP20 = talib.MAROCP(timeperiod=20)
        self.MAROCP30 = talib.MAROCP(timeperiod=30)
        self.MAROCP60 = talib.MAROCP(timeperiod=60)
        self.MAROCP90 = talib.MAROCP(timeperiod=90)

        
        #
        self.MARelative5 = talib.MARelative(timeperiod=5)
        self.MARelative10 = talib.MARelative(timeperiod=10)
        self.MARelative20 = talib.MARelative(timeperiod=20)
        self.MARelative30 = talib.MARelative(timeperiod=30)
        self.MARelative60 = talib.MARelative(timeperiod=60)
        self.MARelative90 = talib.MARelative(timeperiod=90)

        
        #VMA
        #
        self.VolumeRelative5 = talib.VolumeRelative(timeperiod=5)
        self.VolumeRelative10 = talib.VolumeRelative(timeperiod=10)
        self.VolumeRelative20 = talib.VolumeRelative(timeperiod=20)
        self.VolumeRelative30 = talib.VolumeRelative(timeperiod=30)
        self.VolumeRelative60 = talib.VolumeRelative(timeperiod=60)
        self.VolumeRelative90 = talib.VolumeRelative(timeperiod=90)
    
        
        # price-volume
        self.PriceVolume = talib.PriceVolume()
        
        
        # extra added
        self.KDJ = talib.KDJRelated(fastk_period=9, slowk_period=3, slowd_period=3)
        
        # journal
        self.ReturnsRollingStd4 = jf.ReturnsRollingStd(window=4)
        self.ReturnsRollingStd12 = jf.ReturnsRollingStd(window=12)
        self.ReturnsRollingStd22 = jf.ReturnsRollingStd(window=22)
        
        self.BackwardSharpRatio4 = jf.BackwardSharpRatio(window=4)
        self.BackwardSharpRatio12 = jf.BackwardSharpRatio(window=12)
        self.BackwardSharpRatio22 = jf.BackwardSharpRatio(window=22)
        self.BackwardSharpRatio36 = jf.BackwardSharpRatio(window=36)
        
        # trading factor
        self.VolumeReturnsCorr4 = vp.VolumeReturnsCorr(window=4)
        self.VolumeReturnsCorr12 = vp.VolumeReturnsCorr(window=12)
        self.VolumeReturnsCorr22 = vp.VolumeReturnsCorr(window=22)
        
        self.HighLowCorr4 = vp.HighLowCorr(window=4)
        self.HighLowCorr12 = vp.HighLowCorr(window=12)
        self.HighLowCorr22 = vp.HighLowCorr(window=22)
        
        
        self.VolumeVwapDeviation4 = vp.VolumeVwapDeviation(window=4)
        self.VolumeVwapDeviation12 = vp.VolumeVwapDeviation(window=12)
        self.VolumeVwapDeviation22 = vp.VolumeVwapDeviation(window=22)
        
        self.OpenJump = vp.OpenJump()
        self.AbnormalVolume4 = vp.AbnormalVolume(window=4)
        self.AbnormalVolume12 = vp.AbnormalVolume(window=12)
        self.AbnormalVolume22 = vp.AbnormalVolume(window=22)
        
        self.VolumeRangeDeviation4 = vp.VolumeRangeDeviation(window=4)
        self.VolumeRangeDeviation12 = vp.VolumeRangeDeviation(window=12)
        self.VolumeRangeDeviation22 = vp.VolumeRangeDeviation(window=22)
        
    def forward(self, open_ts, high_ts, low_ts, close_ts, volume_ts):
        feature_list = []
        
        # data
        returns_ts = self.pct_change(close_ts)
        # 4
        rocp = self.ROCP(close_ts)
        orocp = self.ROCP(open_ts)
        hrocp = self.ROCP(high_ts)
        lrocp = self.ROCP(low_ts)
        
        feature_list.extend([rocp, orocp, hrocp, lrocp])
        
        # 6
        norm_DIF, norm_DEA, norm_MACD, norm_DIF_diff, norm_DEA_diff, norm_MACD_diff = self.MACD(close_ts)
        feature_list.extend([norm_DIF, norm_DEA, norm_MACD, norm_DIF_diff, norm_DEA_diff, norm_MACD_diff])
        
        # 6
        RSI6 = self.RSI6(close_ts)
        RSI12 = self.RSI12(close_ts)
        RSI24 = self.RSI24(close_ts)
        
        RSIROCP6 = self.RSIROCP6(close_ts)
        RSIROCP12 = self.RSIROCP12(close_ts)
        RSIROCP24 = self.RSIROCP24(close_ts)
        feature_list.extend([RSI6,RSI12,RSI24,RSIROCP6,RSIROCP12,RSIROCP24])
        
        # 1
        VolumeROCP = self.VROCP(volume_ts)
        feature_list.extend([VolumeROCP])
        
        # 3
        upperband_relative_ts, middleband_relative_ts, lowerband_relative_ts = self.BOLL(close_ts)
        feature_list.extend([upperband_relative_ts, middleband_relative_ts, lowerband_relative_ts])
        
        # 10
        MAROCP5= self.MAROCP5(close_ts)
        MAROCP10= self.MAROCP10(close_ts)
        MAROCP20= self.MAROCP20(close_ts)
        MAROCP30= self.MAROCP30(close_ts)
        MAROCP60= self.MAROCP60(close_ts)
        MAROCP90= self.MAROCP90(close_ts)

        # 10
        MARelative5= self.MARelative5(close_ts)
        MARelative10= self.MARelative10(close_ts)
        MARelative20= self.MARelative20(close_ts)
        MARelative30= self.MARelative30(close_ts)
        MARelative60= self.MARelative60(close_ts)
        MARelative90= self.MARelative90(close_ts)

        
        feature_list.extend([MAROCP5,MAROCP10,MAROCP20,MAROCP30,MAROCP60,MAROCP90])
        feature_list.extend([MARelative5,MARelative10,MARelative20,MARelative30,MARelative60,MARelative90])
        
        # 10 VMAROCP
        VMAROCP5= self.MAROCP5(volume_ts)
        VMAROCP10= self.MAROCP10(volume_ts)
        VMAROCP20= self.MAROCP20(volume_ts)
        VMAROCP30= self.MAROCP30(volume_ts)
        VMAROCP60= self.MAROCP60(volume_ts)
        VMAROCP90= self.MAROCP90(volume_ts)

        # 10 Vma relative
        VolumeRelative5= self.VolumeRelative5(volume_ts)
        VolumeRelative10= self.VolumeRelative10(volume_ts)
        VolumeRelative20= self.VolumeRelative20(volume_ts)
        VolumeRelative30= self.VolumeRelative30(volume_ts)
        VolumeRelative60= self.VolumeRelative60(volume_ts)
        VolumeRelative90= self.VolumeRelative90(volume_ts)

       
        feature_list.extend([VMAROCP5,VMAROCP10,VMAROCP20,VMAROCP30,VMAROCP60,VMAROCP90])#, VMAROCP360, VMAROCP720])
        feature_list.extend([VolumeRelative5,VolumeRelative10,VolumeRelative20,VolumeRelative30,VolumeRelative60,VolumeRelative90])#, VolumeRelative360, VolumeRelative720])
       
        # price_volume
        PriceVolume = self.PriceVolume(close_ts, volume_ts)
        feature_list.extend([PriceVolume])
        
        # dkj
        RSV, K, D, J = self.KDJ(high_ts, low_ts, close_ts)
        feature_list.extend([RSV, K, D, J])
        
        # journalhub
        ReturnsRollingStd4 = self.ReturnsRollingStd4(returns_ts)
        ReturnsRollingStd12 = self.ReturnsRollingStd12(returns_ts)
        ReturnsRollingStd22 = self.ReturnsRollingStd22(returns_ts)
        
        BackwardSharpRatio4 = self.BackwardSharpRatio4(returns_ts)
        BackwardSharpRatio12 = self.BackwardSharpRatio12(returns_ts)
        BackwardSharpRatio22 = self.BackwardSharpRatio22(returns_ts)
        BackwardSharpRatio36 = self.BackwardSharpRatio36(returns_ts)
        
        feature_list.extend([ReturnsRollingStd4,ReturnsRollingStd12,ReturnsRollingStd22,BackwardSharpRatio4,BackwardSharpRatio12,BackwardSharpRatio22, BackwardSharpRatio36])
        #
        
        VolumeReturnsCorr4 = self.VolumeReturnsCorr4(volume_ts, returns_ts)
        VolumeReturnsCorr12 = self.VolumeReturnsCorr12(volume_ts, returns_ts)
        VolumeReturnsCorr22 = self.VolumeReturnsCorr22(volume_ts, returns_ts)
        
        HighLowCorr4 = self.HighLowCorr4(high_ts, low_ts)
        HighLowCorr12 = self.HighLowCorr12(high_ts, low_ts)
        HighLowCorr22 = self.HighLowCorr22(high_ts, low_ts)
        
        
        VolumeVwapDeviation4 = self.VolumeVwapDeviation4(close_ts, volume_ts)
        VolumeVwapDeviation12 = self.VolumeVwapDeviation12(close_ts, volume_ts)
        VolumeVwapDeviation22 = self.VolumeVwapDeviation22(close_ts, volume_ts)
        feature_list.extend([VolumeReturnsCorr4,VolumeReturnsCorr12,VolumeReturnsCorr22,HighLowCorr4,HighLowCorr12,HighLowCorr22,VolumeVwapDeviation4,VolumeVwapDeviation12,VolumeVwapDeviation22])
        
        OpenJump = self.OpenJump(open_ts, close_ts)
        feature_list.extend([OpenJump])
        
        AbnormalVolume4 = self.AbnormalVolume4(volume_ts)
        AbnormalVolume12 = self.AbnormalVolume12(volume_ts)
        AbnormalVolume22 = self.AbnormalVolume22(volume_ts)
        
        VolumeRangeDeviation4 = self.VolumeRangeDeviation4(high_ts,low_ts,volume_ts)
        VolumeRangeDeviation12 = self.VolumeRangeDeviation12(high_ts,low_ts,volume_ts)
        VolumeRangeDeviation22 = self.VolumeRangeDeviation22(high_ts,low_ts,volume_ts)
        feature_list.extend([AbnormalVolume4,AbnormalVolume12,AbnormalVolume22,VolumeRangeDeviation4,VolumeRangeDeviation12,VolumeRangeDeviation22])
        # label
        label = self.labeler(close_ts)
        feature_list.extend([label])
        return feature_list

if __name__ == "__main__":
    import torch
    order_book_ids = 10
    sequence_len = 100
    
    open_ts = abs(torch.randn((sequence_len, order_book_ids)))
    high_ts = abs(torch.randn((sequence_len, order_book_ids)))
    low_ts = abs(torch.randn((sequence_len, order_book_ids)))
    close_ts = abs(torch.randn((sequence_len, order_book_ids)))
    volume_ts = abs(torch.randn((sequence_len, order_book_ids)))
    
    featurizer = DefaultFeaturizer()
    feature_list = featurizer.forward(open_ts, high_ts, low_ts, close_ts, volume_ts)
    print(len(feature_list))
    

