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
import time
import datetime
import logging
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine, Events, State, _prepare_batch, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import EarlyStopping, ModelCheckpoint
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError('no tensorboardX package found')
from transformerquant.utils.datetime_converter import _to_hours_mins_secs

logger = logging.getLogger(__name__)

class State(object):
    """An object that is used to pass internal and user-defined state between event handlers."""
    def __init__(self, **kwargs):
        self.epoch = 0
        self.iteration = 0
        self.output = None
        self.batch = None
        for k, v in kwargs.items():
            setattr(self, k, v)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
    if isinstance(m, (nn.LSTMCell, nn.LSTM)):
        lstm_weights(m)

def lstm_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_normal_(param)

def create_own_trainer(model, optimizer, loss_fn, metrics={}, device=torch.device('cpu'), clip=False):

    def _update(engine, batch):
        model.train()
        x, y = _prepare_batch(batch, device)
        y_pred = model(x)
        #pdb.set_trace()
        loss = loss_fn(y_pred, y)
        #custom
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if clip:
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)

        return {'loss':loss.item(),
                'y_pred':y_pred,
                'y':y}

    engine =  Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_own_evaluator(model, metrics, device):
    """
    Factory function for creating an evaluator for supervised models
    Args:
        model (torch.nn.Module): the model to train
        metrics (dict of str: Metric): a map of metric names to Metrics
        cuda (bool, optional): whether or not to transfer batch to GPU (default: False)
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    def _inference(engine, batch):
        model.eval()
        x, y = _prepare_batch(batch, device)
        y_pred = model(x)
        return y_pred, y

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_summary_writer(model, log_dir, data_loader):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
    return writer


def output_transform(output):
    y_pred = output['y_pred']
    y = output['y']
    return y_pred, y


def score_function(engine):
    loss_val = engine.state.metrics['loss']
    return -loss_val



def create_task_metrics(torch_dtype, loss_func, output_transform=lambda x:x):
    if torch_dtype in (torch.float32, torch.float64):
        metrics = {'loss':Loss(loss_func, output_transform=output_transform)}

    elif torch_dtype in (torch.int16, torch.int32, torch.int64):
        metrics={'accuracy': Accuracy(output_transform=output_transform),
                 'precision': Precision(output_transform=output_transform),
                 'recall': Recall(output_transform=output_transform),
                 'loss':Loss(F.nll_loss, output_transform=output_transform)}
    return metrics



class Agent(object):
    """agent for train SL and RL model"""

    def __init__(self,
                 model,
                 use_cuda=False,
                 loss_func = None,
                 optimizer=None,
                 n_epochs=10,
                 **kwargs):
        #
        self._trainer = None
        self._evaluator = None
        #
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        #
        self.model = model
        self.model_name = model.model_name if hasattr(model, "model_name") else model.__class__.__name__
        self.model.apply(weights_init)
        self.model = self.model.to(self.device)

        #
        self.n_epochs = n_epochs
        self.early_stop_patience =kwargs.pop('early_stop_patience', 20)
        
        #loss function, optimizer and lr_scheduler
        self.loss_func = loss_func if loss_func is not None else F.nll_loss
        self.optimizer = optimizer
        self.lr_scheduler = kwargs.pop('lr_scheduler', None) #
        self.clip_gradient = kwargs.pop('clip_gradient', False)
        
        
        # log dir and to_save_dir
        default_save_dir = os.path.join(os.path.expanduser("~"), "temp")
        to_save_dir = kwargs.pop('to_save_dir', default_save_dir)
        self.log_interval = kwargs.pop('log_interval', 5)
        self.log_dir = kwargs.pop('log_dir', os.path.join(to_save_dir,'log'))
        now = datetime.datetime.now()
        self.to_save_dir = os.path.join(to_save_dir, "training_{}_{}".format(self.model_name, now.strftime("%Y%m%d_%H%M%S")))
        logger.info("the dir to save model is {}".format(self.to_save_dir))
        #
        checkpoint = kwargs.pop('checkpoint', None)
        if checkpoint is not None:
            logger.info("model is to load checkpoint...")
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            logger.info("model loaded checkpoint successfully")
        #pdb.set_trace()

    def set_trainer(self, trainer=None):
        assert isinstance(trainer, Engine), "trainer is instance of ignite Engine"
        self._trainer = trainer

    def set_evaluator(self, evaluator):
        assert isinstance(evaluator, Engine), "trainer is instance of ignite Engine"
        self._evaluator = evaluator

    def predict(self, dataset, to_evaluate=True):
        self.model.eval()

        if isinstance(dataset, torch.Tensor):
            feature_ts = dataset.to(self.device)
            to_evaluate = False
        elif isinstance(dataset, torch.utils.data.Dataset):
            feature_ts = dataset.feature_ts.to(self.device)
            target_ts = dataset.target_ts
            dataset = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        elif isinstance(dataset, torch.utils.data.DataLoader):
            feature_ts = dataset.dataset.feature_ts.to(self.device)
            target_ts = dataset.dataset.target_ts
        else:
            raise KeyError("input must be torch.utils.data.Dataset or DataLoader")

        y_pred = self.model(feature_ts)

        if to_evaluate:
            evaluator = self._evaluator
            if evaluator is None:
                self._logger.info("loading default evaluator in prophet")
                evaluator = self.create_default_evaluator(dataset)

            @evaluator.on(Events.EPOCH_COMPLETED)
            def log_training_metric(engine):
                for metric_name in engine.state.metrics.keys():
                    value = engine.state.metrics[metric_name]
                    #avg_value = torch.mean(value)
                    print("Result on test dataset {}:{}".format(metric_name, value))
            state = evaluator.run(dataset, max_epochs=1)

        return y_pred

    def fit(self, loader_train, loader_val=None):
        # set the model in train mode
        self.model.train()

        #
        trainer = self._trainer
        if trainer is None:
            logger.info("loading default trainer in prophet")
            trainer = self.create_default_trainer(loader_train)
        if loader_val is not None:
            evaluator = self._evaluator
            if evaluator is None:
                logger.info("loading default evaluator in prophet")
                evaluator = self.create_default_evaluator(loader_val)

        trainer = self._register_train_events(loader_train, loader_val, trainer, evaluator)
        state = trainer.run(loader_train, max_epochs=self.n_epochs)
        return state

    def create_default_trainer(self, dataloader_to_exec):
        """you can override this method by create your own trainer and evaluator"""
        label_dtype = dataloader_to_exec.dataset[0][-1].dtype#.target_ts.dtype
        metrics_train = create_task_metrics(label_dtype, self.loss_func, output_transform)
        trainer = create_own_trainer(self.model, self.optimizer, self.loss_func, metrics = metrics_train, device=self.device, clip=self.clip_gradient)
        return trainer

    def create_default_evaluator(self, dataloader_to_eval):
        label_dtype = dataloader_to_eval.dataset[0][-1].dtype#.target_ts.dtype
        metrics_eval = create_task_metrics(label_dtype, self.loss_func)
        evaluator = create_supervised_evaluator(self.model, metrics=metrics_eval, device=self.device)
        return evaluator

    def _register_train_events(self, dataloader_to_exec, dataloader_to_eval=None, trainer=None, evaluator=None):
        writer = create_summary_writer(self.model, self.log_dir, dataloader_to_exec)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter = (engine.state.iteration - 1) % len(dataloader_to_exec) + 1
            if iter % self.log_interval == 0:
                print("Epoch[{}], Iteration[{}/{}], loss:{:.4f}".format(engine.state.epoch, iter, len(dataloader_to_exec), engine.state.output['loss']))
                writer.add_scalar('training/loss', engine.state.output['loss'], engine.state.iteration)

        @trainer.on(Events.EPOCH_STARTED)
        def update_lr_scheduler(engine):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_metric(engine):
            for metric_name in engine.state.metrics.keys():
                value = engine.state.metrics[metric_name]
                #avg_value = torch.mean(value)
                print("Epoch[{}], Epoch End {}:{}".format(engine.state.epoch, metric_name, value))


        if (dataloader_to_eval is not None) and (evaluator is not None):

            @trainer.on(Events.EPOCH_COMPLETED)
            def log_validation_results(engine):
                state_val = evaluator.run(dataloader_to_eval, max_epochs=1)
                engine.state.state_val = state_val
                for metric_name in state_val.metrics.keys():
                    value = state_val.metrics[metric_name]
                    print("Validation Results - Epoch: {}  {}: {}".format(engine.state.epoch, metric_name, value))
                writer.add_scalar('validation/loss', evaluator.state.metrics['loss'], engine.state.epoch)

            #setup early stopping
            handler = EarlyStopping(patience=self.early_stop_patience, score_function=score_function, trainer=trainer)
            evaluator.add_event_handler(Events.COMPLETED, handler)

            #setup model checkpoint
            saver_best_model = ModelCheckpoint(dirname =self.to_save_dir,
                                               filename_prefix='model',
                                               score_name='val_loss',
                                               score_function = score_function,
                                               n_saved=2,
                                               atomic=True,
                                               create_dir=True,
                                               save_as_state_dict=True)
            evaluator.add_event_handler(Events.COMPLETED, saver_best_model, {self.model_name:self.model})

        @trainer.on(Events.COMPLETED)
        def close_writer(engine):
            writer.close()

        return trainer


class AgentBase(object):
    """agent for train SL and RL model"""

    def __init__(self,
                 model,
                 use_cuda=False,
                 loss_func = None,
                 optimizer=None,
                 n_epochs=10,
                 **kwargs):
        
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        #
        self.model = model
        self.model_name = model.__class__.__name__
        self.model.apply(weights_init)
        self.model = self.model.to(self.device)

        #
        self.n_epochs = n_epochs
        self.early_stop_patience =kwargs.pop('early_stop_patience', 20)

        #
        #loss function, optimizer and lr_scheduler
        self.loss_func = loss_func if loss_func is not None else F.nll_loss
        self.optimizer = optimizer
        self.lr_scheduler = kwargs.pop('lr_scheduler', None) #
        self.clip_gradient = kwargs.pop('clip_gradient', False)
        # log dir and to_save_dir
        default_save_dir = os.path.join(os.path.expanduser("~"), "temp")
        to_save_dir = kwargs.pop('to_save_dir', default_save_dir)
        self.log_interval = kwargs.pop('log_interval', 5)
        self.log_dir = kwargs.pop('log_dir', os.path.join(to_save_dir,'log'))
        now = datetime.datetime.now()
        self.to_save_dir = os.path.join(to_save_dir, "training_{}_{}".format(self.model_name, now.strftime("%Y%m%d_%H%M%S")))
        logger.info("the dir to save model is {}".format(self.to_save_dir))
        #
        checkpoint = kwargs.pop('checkpoint', None)
        if checkpoint is not None:
            logger.info("model is to load checkpoint...")
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
            logger.info("model loaded checkpoint successfully")
    
    def fit(self, dataloader_train, dataloader_val=None):
        
        self.state = State(max_epochs=self.n_epochs, metrics={})
        for epoch in range(self.n_epochs):
            self.state.epoch += 1
            self.model.train()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            hours, mins, secs = self._run_once_on_dataset(dataloader_train)
            # evaluate val datalooader and save best model
            if dataloader_val is not None:
                loss = self.evaluate_dataloader(dataloader_val)
                self._save_model(-loss)
        return self.state
    
    
    def _run_once_on_dataset(self, dataloader):
        start_time = time.time()
        loss_list = []
        try:
            for i, batch in enumerate(dataloader):
                #1 get the inputs
                inputs, labels = batch
                    
                #2 wrap them in device (Variabel)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                #3 forward pass: compute predict y by passing x to the model
                y_pred = self.model(inputs)
                
                #4 compute and print loss
                #pdb.set_trace()
                loss = self.loss_func(y_pred, labels)
                loss_list.append(loss.item())    
                #5 zero gradient, perform a backward propagation and update the weight
                self.optimizer.zero_grad()
                loss.backward()
                if self.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                
                if i % self.log_interval == 0:
                    logger.info("epoch:{}, iteration:{}, loss:{:.5f}".format(self.state.epoch, i, loss.item()))
                
        except Exception as e:
            logger.error("Current run is terminating due to exception: %s.", str(e))
        
        time_taken = time.time() - start_time
        hours, mins, secs = _to_hours_mins_secs(time_taken)
        self.state.loss_train = np.nanmean(loss_list)
        logger.info("Epoch {} Complete. Time taken: {:.0f}:{:.0f}:{:.0f}, loss:{:.5f}".format(self.state.epoch, hours, mins, secs, self.state.loss_train))
        
        
        return hours, mins, secs
    
    
    def evaluate_dataloader(self, dataloader):
        self.model.eval()
        with torch.no_grad(): # to save memory
            correct = 0
            total = 0
            loss_list = []
            for i, batch in enumerate(dataloader):
                #1 get the inputs
                inputs, labels = batch
                
                #2 wrap them in device (Variabel)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                #3 forward pass: compute predict y by passing x to the model
                y_pred = self.model(inputs)
                
                #4 compute and print loss
                #pdb.set_trace()
                loss = self.loss_func(y_pred, labels)
                
                loss_list.append(loss.item())
                if labels.dtype in (torch.int16, torch.int32, torch.int64):
                    total += labels.size(0)
                    correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
            
            loss = np.nanmean(loss_list)
            logger.info("[evaluate mode] loss on dataset:{:.5f}".format(loss))
            self.state.loss_val = loss
            return loss
                
    def predict(self, dataset):
        self.model.eval()

        if isinstance(dataset, torch.Tensor):
            feature_ts = dataset.to(self.device)
        elif isinstance(dataset, torch.utils.data.Dataset):
            feature_ts = dataset.feature_ts.to(self.device)
        elif isinstance(dataset, torch.utils.data.DataLoader):
            feature_ts = dataset.dataset.feature_ts.to(self.device)
        else:
            raise KeyError("input must be torch.utils.data.Dataset or DataLoader")
        y_pred = self.model(feature_ts)
        return y_pred
    
    def _save_model(self, score):
        if not getattr(self, "best_score", None):
            self.best_score = 0
        if getattr(self, 'to_save_dir', None):
            if not os.path.exists(self.to_save_dir):
                os.makedirs(self.to_save_dir)
        if score >= self.best_score:
            try:
                os.remove(os.path.join(self.to_save_dir, '{}_{}.pth'.format(self.model_name, self.best_score)))
            except OSError:
                pass
            self.best_score = score
            torch.save(self.model.state_dict(), os.path.join(self.to_save_dir, '{}_{}.pth'.format(self.model_name, self.best_score)))
            logger.info("saveing best model")