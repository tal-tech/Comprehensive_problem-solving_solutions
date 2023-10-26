#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
from procedure_model.src.procedure_utils import *

# 循环小数
def clean_rotate_number(s):
    s = s.replace('¯','')
    s = s.replace(' ','')
    s = s.replace(':','')
    s = s.replace('\quad','')
    return s
def rule_rotate_number(stuAns_row, procedure, thres):
    decide = 0
    if (('<' in stuAns_row) or ('>' in stuAns_row)):
        decide = 0
    elif (procedure in clean_rotate_number(stuAns_row)):
        decide = 1
    else:
        decide = 0
    return decide


