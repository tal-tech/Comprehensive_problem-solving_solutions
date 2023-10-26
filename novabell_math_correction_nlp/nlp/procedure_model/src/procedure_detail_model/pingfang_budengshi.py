#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy
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

def calculate_similarity_B010(procedure_list, stu_answer_row):
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
        list_procedure = extract_operator_and_integer(procedure)
        list_ans = extract_operator_and_integer(stu_answer_row)
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
def rule_pingfangbudengshi(stuAns_row, procedure,thres):
    '''
    stuAns: 
        type: str 
        sample: 'I like food\nyou like what\n'
    procedure_list:
        type: list
        sample: ['Frist try this', 'Then try this']
    '''
    procedure_list = extend_B010([procedure])
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_B010(procedure_list,row) >=thres:
            decide =1
    return decide


