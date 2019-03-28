#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 26 March 2019 16:29
from templates import train_template
from transformer2.model import Transformer


train_template(Transformer, shuffle=False, save_model=True)
