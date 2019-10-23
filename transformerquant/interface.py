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

import abc

class AbstractConfig(object):
    """Abstract configuration class"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Instantiate a :class:`PretrainedConfig` (or a derived class) from a pre-trained model configuration."""
        raise NotImplementedError
        
    @abc.abstractmethod
    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`transformerquant.PretrainedConfig.from_pretrained` class method.
        """
        raise NotImplementedError


class AbstractModel(abc.ABCMeta):
    """Abstract class for all models"""
    
    @abc.abstractmethod
    def from_pretrained(self, pretrained_model_name_or_path, **kwargs):
        """Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~transformerquant.PreTrainedModel.from_pretrained`` class method.
        """
        raise NotImplementedError
        
    @abc.abstractmethod
    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`transformerquant.PreTrainedModel.from_pretrained`` class method.
        """
        raise NotImplementedError

    