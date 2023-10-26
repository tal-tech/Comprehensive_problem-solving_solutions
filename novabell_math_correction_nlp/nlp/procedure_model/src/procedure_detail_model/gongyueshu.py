#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import numpy as np
from functools import reduce
import math
from itertools import permutations
from procedure_model.src.procedure_utils import *

# 公约数
def get_gcd_for(list_):
    return reduce(math.gcd,list_)
def extract_integer(text):
    return re.findall(r'-?\d+', text)
def perm(s):
    s1 = []
    for item in permutations(s):
        t = []
        for a in item:
            t.append(a)
        s1.append(t)
    return s1
def expand_gongyueshu(procedure):
    pro = clean(procedure)
    new_pro = pro
    new_list = []
    if re.search(r'\(\d+[\s\S]*\d+\)=\d+',pro):
        exist_num=True
        lnum = re.findall(r'\((\d+)', pro)[0]
        rnum = re.findall(r'(\d+)\)', pro)[0]
        new_list.append('('+str(lnum)+','+str(rnum)+')')
        new_list.append('('+str(rnum)+','+str(lnum)+')')
        new_list.append('['+str(lnum)+','+str(rnum)+']')
        new_list.append('['+str(rnum)+','+str(lnum)+']')
        new_list.append('{'+str(lnum)+','+str(rnum)+'}')
        new_list.append('{'+str(rnum)+','+str(lnum)+'}')
        temp_list = [pro]
        temp_list = [x.split('=')[0] for x in temp_list if '=' in x]
        if len(temp_list) > 0:
            temp_list = list(set(np.hstack([extract_integer(x) for x in temp_list])))
            new_list.append(','.join([x for x in temp_list if x != lnum and x != rnum]))
            new_list.append(','.join([lnum, rnum]))
        return [x for x in list(set([pro] + new_list)) if x != '']
def expand_gongyueshu_new(procedure):
    pro = clean(procedure)
    new_pro = pro
    new_list = []
    temps = re.findall(r'\(\d+[,;、，；]+\d+\)',pro)
    for temp in temps:
        nums = re.findall(r'\d+', temp)
        extended_nums = perm(nums)
        for arr in extended_nums:
            new_list.append(pro.replace(temp, ','.join(arr)+'的公约数'))
            new_list.append(pro.replace(temp, '('+','.join(arr)+')'))
            new_list.append(pro.replace(temp, '['+','.join(arr)+']'))
            new_list.append(pro.replace(temp, '{'+','.join(arr)+'}'))
    return [x for x in list(set([pro] + new_list)) if x != '']
def extract_operator_and_number_gongyueshu(s):
    return re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', s) + re.findall(r'\d+(?:\\d+)?', s)
def calculate_similarity_gongyueshu(procedure_list, stu_answer_row):
    '''
    procedure_list:
        type: list
        sample: ['1+1=2','2-1=1']
    stu_answer_row:
        type: str
        sample: '1+1=2'
    '''
    tmp = []
    for procedure in procedure_list:
        list_procedure = extract_operator_and_number_gongyueshu(procedure)
        list_ans = extract_operator_and_number_gongyueshu(stu_answer_row)
        intersection = len(set(list_procedure).intersection(list_ans))
        union = len(set(list_procedure)) + len(set(list_ans)) - intersection
        if union == 0:
            list_procedure = [x for x in procedure]
            list_ans = [x for x in stu_answer_row]
            intersection = len(set(list_procedure).intersection(list_ans))
            union = len(set(list_procedure)) + len(set(list_ans)) - intersection
        jaccard = intersection/union
        tmp.append(jaccard)
    return max(tmp)
def rule_gongyueshu(stuAns_row, procedure, thres):
    # procedure_list = expand_gongyueshu(procedure)
    procedure_list = expand_gongyueshu_new(procedure)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity(procedure_list,row)>= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    return decide


