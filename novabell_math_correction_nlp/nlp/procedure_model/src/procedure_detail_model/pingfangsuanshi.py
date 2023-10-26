#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy
import math
from procedure_model.src.procedure_utils import *

def extend_B010(procedure_list):
    new_list = []
    procedure_list = [clean(i) for i in procedure_list]
    for procedure in procedure_list:
        pro = copy.copy(procedure)
        a_pro = copy.copy(pro)
        new_list.append(procedure)
        fanshi = re.findall(r'[\d]+\s*\^\s*[\d]+',procedure)
        this_procedure_change = {}
        for fan in fanshi:
            fanhou = re.findall(r'(?<=\^)\d+',fan)[0]
            fanqian =re.findall(r'[\d]+(?=\^)',fan)[0]
            new_fan = ((fanqian+'*') * int(fanhou))[:-1]
            this_procedure_change[fan] = new_fan
        for i in this_procedure_change:
            pro = pro.replace(i,this_procedure_change[i])
        new_list.append(pro)
        if ('<' in procedure) or ('>' in procedure):
            for i in this_procedure_change:
                a_pro = a_pro.replace(i, str(eval(this_procedure_change[i])))
        new_list.append(a_pro)
    return list(set(new_list))

def rule_pingfang_suanshi(stuAns_row, procedure, thres):
    procedure_list = [clean(procedure)]
    procedure_list = extend_B010(procedure_list)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity(procedure_list, row) >= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    if decide == 1:
        this_row = row.replace('x','*')
        this_row = re.sub('[a-zA-Z]','',this_row)
        this_row = this_row.replace('{','')
        this_row = this_row.replace('}','')
        if '^' in this_row:
            pow_cal = re.findall(r'\d+\^\d+',this_row)
            if pow_cal:
                time = re.findall(r'(?<=\^)\d+',pow_cal[0])
                base = re.findall(r'\d+(?=\^)',pow_cal[0])
                pow_res = int(math.pow(int(base[0]), int(time[0])))
                shizi = this_row.replace(pow_cal[0],str(pow_res))
                shizi = shizi.replace(' ','')
                shizi = re.findall(r'\d+(?:[\+,\-,\*,\/]\d+)*\=\d+(?:[\+,\-,\*,\/]\d+)+',shizi)
                if len(shizi) > 0:
                    shizi = shizi[0]
                    equal_left = shizi[:shizi.find('=')]
                    equal_right = shizi[shizi.find('=')+1:]
                    if eval(equal_left) != eval(equal_right):
                        decide = 0
    return decide

