#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 6/3/2019 3:04 PM
from templates import train_template
from fm.model import FM


train_template(FM, shuffle=True)
