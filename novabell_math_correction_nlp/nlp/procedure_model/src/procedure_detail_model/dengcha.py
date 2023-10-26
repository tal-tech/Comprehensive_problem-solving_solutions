#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *

# 等差数列
def extend_dengcha(procedure):
    new_list = []
    pro = procedure
    new_pro = pro
    new_list.append(new_pro)
    if 'dots' in pro:
        if '=' in pro:
            equ_index = pro.find('=')
            new_equ = pro[:equ_index]
            whole_number = re.findall(r'\d+',new_equ)
            cdot_index = pro.find('dots')
            pre_number = re.findall(r'\d+',pro[:cdot_index])
            head = eval(pre_number[0])
            cha = eval(pre_number[1])-eval(pre_number[0])
            tail = eval(re.findall(r'\d+',pro[cdot_index:])[0])
            dengcha = [x for x in range(head,tail+cha,cha)]
            else_number_list = [eval(x) for x in whole_number if eval(x) not in dengcha]
            if len(else_number_list)>=1:
                for duoshao in range(1,len(dengcha)-1):
                    new_list.append('+'.join([str(dengcha[j]) for j in range(duoshao)] + ['\cdots+{}'.format(tail)])+'+'+'+'.join([str(x) for x in else_number_list])+pro[equ_index:])
                new_list.append('+'.join([str(j) for j in dengcha])+'+'+'+'.join([str(x) for x in else_number_list])+pro[equ_index:])
                new_list.append('{}'.format(sum(dengcha))+'+'+'+'.join([str(x) for x in else_number_list])+pro[equ_index:])
            else:
                for duoshao in range(1,len(dengcha)-1):
                    new_list.append('+'.join([str(dengcha[j]) for j in range(duoshao)] + ['\cdots+{}'.format(tail)])+pro[equ_index:])
                new_list.append('+'.join([str(j) for j in dengcha])+pro[equ_index:])
        else:
            whole_number = re.findall(r'\d+',pro)
            cdot_index = pro.find('dots')
            pre_number = re.findall(r'\d+',pro[:cdot_index])
            head = eval(pre_number[0])
            cha = eval(pre_number[1])-eval(pre_number[0])
            tail = eval(re.findall(r'\d+',pro[cdot_index:])[0])
            dengcha = [x for x in range(head,tail+cha,cha)]
            else_number_list = [eval(x) for x in whole_number if eval(x) not in dengcha]
            if len(else_number_list) >=1:
                for duoshao in range(1,len(dengcha)-1):
                    new_list.append('+'.join([str(dengcha[j]) for j in range(duoshao)] + ['\cdots+{}'.format(tail)])+'+'+'+'.join([str(x) for x in else_number_list]))
                new_list.append('+'.join([str(j) for j in dengcha])+'='+'{}'.format(sum(dengcha)))
                new_list.append('{}'.format(sum(dengcha))+'+'+'+'.join([str(j) for j in else_number_list])+'='+'{}'.format(sum(dengcha+else_number_list)))
            else:
                for duoshao in range(1,len(dengcha)-1):
                    new_list.append('+'.join([str(dengcha[j]) for j in range(duoshao)] + ['\cdots+{}'.format(tail)]))
                new_list.append('+'.join([str(j) for j in dengcha])+'='+'{}'.format(sum(dengcha)))
    return [clean(x) for x in new_list]
def extract_operator_and_number_dengcha(s):
    return re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', s) + re.findall(r'\d+(?:\\d+)?', s)
def calculate_similarity_dengcha(procedure_list, stu_answer_row):
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
        list_procedure = extract_operator_and_number_dengcha(procedure)
        list_ans = extract_operator_and_number_dengcha(stu_answer_row)
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
def rule_dengcha(stuAns_row, procedure, thres):
    procedure_list = extend_dengcha(procedure)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity_dengcha(procedure_list, row) >= thres:
            decide = 1
        else:
            plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            if extract_operator_and_number(row) in plus_rule:
                decide = 1
    return decide


