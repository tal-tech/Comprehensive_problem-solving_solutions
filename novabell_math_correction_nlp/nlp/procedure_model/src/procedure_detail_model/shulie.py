#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy
import math
from procedure_model.src.procedure_utils import *

def extend_pingfang_shulie(procedure):
    procedure_clean = clean(procedure)
    pro = copy.copy(procedure)
    new_list = []
    new_list.append(procedure_clean)
    chi = re.findall('[\u4E00-\u9FA5]+',pro)
    operator =re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', pro)
    if (not chi) and (not operator):
        num = re.findall(r'\d+',pro)
        new_num = []
        for i in num:
            genhao = math.sqrt(eval(i))
            if genhao - int(genhao) == 0:
                new_num.append(str(int(genhao))+'^2')
        if len(new_num) == len(num):
            new_list.append(','.join(new_num))
    return new_list

def calculate_similarity_shulie(procedure_list, stu_answer_row):
    tmp = []
    for procedure in procedure_list:
        list_procedure = extract_integer(procedure)
        list_ans = extract_integer(stu_answer_row)
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

def rule_pingfang_shulie(stuAns_row, procedure, thres=1):
    procedure_list = extend_pingfang_shulie(procedure)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_shulie(procedure_list, row) >= thres:
            decide = 1
        '''
        else:
            plus_rule = [extract_integer(x) for x in procedure_list]
            if any(x in extract_integer(row) for x in plus_rule):
                decide = 1
        '''
    return decide


