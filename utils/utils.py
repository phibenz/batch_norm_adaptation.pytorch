from __future__ import division

import os, time, random, json, logging 
from datetime import datetime
import numpy as np
import torch

from config.config import RESULT_PATH, DATA_PATH

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def get_timestamp():
    ISOTIMEFORMAT='%Y%m%d_%H%M%S_%f'
    timestamp = '{}'.format(datetime.utcnow().strftime( ISOTIMEFORMAT)[:-3])
    return timestamp

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d_%H+%M+%S'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def time_file_str():
    ISOTIMEFORMAT='%Y-%m-%d'
    string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string + '-{}'.format(random.randint(1, 10000))

def get_model_path(dataset_name, network_arch):
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    model_id = "{}_{}".format(dataset_name, network_arch)
    model_path = os.path.join(RESULT_PATH, model_id)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    return model_path

def save_config(path, args):
    with open(os.path.join(path, "config.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def load_config(path):
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)

def get_logger(path):
    logger = logging.getLogger('logbuch')
    logger.setLevel(level=logging.DEBUG)
    
    # Stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    sh.setFormatter(sh_formatter)
    
    # File handler
    fh = logging.FileHandler(os.path.join(path, "log.txt"))
    fh.setLevel(level=logging.DEBUG)
    fh_formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
    fh.setFormatter(fh_formatter)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)
