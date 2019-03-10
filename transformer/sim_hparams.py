#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 7:29 PM

from transformer.hparams import Hparams


class SimHparams(Hparams):
    def __init__(self):
        super(SimHparams, self).__init__()
        self.parser.add_argument('--num_hidden_layers', default=1, type=int)
        self.parser.add_argument('--hidden_size', default=512, type=int)
        self.parser.add_argument('--dropout', default=0.3, type=float)
        self.parser.add_argument('--train_sim', default="./data/sim/sim.dat")
