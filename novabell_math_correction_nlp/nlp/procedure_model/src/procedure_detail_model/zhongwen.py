#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *

def rule_chinese(stuAns_row, procedure, thres):
    pro = clean(procedure)
    decide = 0
    row = clean(stuAns_row)
    if pro in row:
        decide = 1
    elif calculate_chinese_similarity(pro, row) >= thres:
        decide = 1
    return decide