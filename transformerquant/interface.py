#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

    