#hparam.py

import os

class Hyperparameters:
    check_mode = False
    
    max_epoch = 1000
    min_epoch = 100
    batch_size = 64
    num_workers = 8
    test_size = 0.2
    random_state = 42
    patience = 10
    hidden_channels = [1024, 1024, 1024, 2048, 2048, 2048]
    learning_rate = 0.001
    weight_decay = 0.001 
    lr_patience = 200
    lr_factor = 0.8
    dropout = 0.1