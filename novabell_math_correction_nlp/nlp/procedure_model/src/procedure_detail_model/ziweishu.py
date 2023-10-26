#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy
from procedure_model.src.procedure_utils import *

def calculate_similarity_B008(procedure_list, stu_answer_row):
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
        list_procedure = extract_chi_number(procedure)
        list_ans = extract_chi_number(stu_answer_row)
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
def rule_ziweishu(stuAns_row, procedure,thres=0.53):
    '''
    stuAns: 
        type: str 
        sample: 'I like food\nyou like what\n'
    procedure_list:
        type: list
        sample: ['Frist try this', 'Then try this']
    '''
    procedure_list = [clean(procedure)]
    decide = 0
    procedure_list = extend_B008(procedure_list)
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_B008(procedure_list,row) >=thres:
            decide = 1
        else:
            plus_rule = [extract_chi_number(x) for x in procedure_list]
            if extract_chi_number(row) in plus_rule:
                decide = 1
    return decide

def extend_B008(procedure_list):
    new_list = []
    for procedure in procedure_list:
        pro = copy.copy(procedure)
        new_list.append(procedure)
        if '为' in procedure:
            new_list.append(procedure.replace('为','='))
        fanshi = re.findall(r'[\u4E00-\u9FA5]+\*[\d]+',procedure)
        this_procedure_change = {}
        for fan in fanshi:
            times = re.findall(r'(?<=\*)\d+',fan)[0]
            wenzi =re.findall(r'[\u4E00-\u9FA5]+(?=\*)',fan)[0]
            this_procedure_change[fan] = ((wenzi+'+')*int(times))[:-1]
        for i in this_procedure_change:
            pro = pro.replace(i,this_procedure_change[i])
        new_list.append(pro)
    return list(set(new_list))













