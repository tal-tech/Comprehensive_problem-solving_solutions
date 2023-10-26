#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import copy
from procedure_model.src.procedure_utils import *

def extend_B012(procedure_list):
    new_list = []
    procedure_list = [clean(i) for i in procedure_list]
    for procedure in procedure_list:
        new_list.append(procedure)
        pro = copy.copy(procedure)
        need_add = {}
        shu_times = re.findall(r'\d+\s+\*\s+\d+',procedure)
        zi_times = re.findall(r'\d+\s+\*\s+[a-zA-Z]+',procedure) + re.findall(r'[a-zA-Z]+\s+\*\s+\d+',procedure)
        for i in shu_times:
            need_add[i] = str(eval(i))
        for i in zi_times:
            keke_i = i.replace(' ','')
            shu = re.findall(r'\d+',keke_i)
            zi = re.findall(r'[a-zA-Z]+',keke_i)
            need_add[i] = shu[0]+zi[0]
        for i in need_add:
            pro = pro.replace(i,need_add[i])
        new_list.append(pro)
    return list(set(new_list))
def huansuan_fuhao(procedure_list):
    new_list = []
    procedure_list = [i.replace(' ','') for i in procedure_list]
    for pro in procedure_list:
        new_list.append(pro)
        new_pro = copy.copy(pro)
        not_shuzi_zimu = re.findall(r'[^a-zA-Z0-9]+',pro)
        if all(x in ['+','='] for x in not_shuzi_zimu):
            ele = re.findall(r'[\da-zA-Z]+',pro)
            if len(ele) == 3:
                equal_index = pro.find('=')
                equal_left = pro[:equal_index]
                equal_right = pro[equal_index+1:]
                left_jia = re.findall('\+[\da-zA-Z]+',equal_left)
                for i in left_jia:
                    new_pro = new_pro.replace(i,'')
                    new_pro+=i.replace('+','-')
                new_list.append(new_pro)
    return list(set(new_list))

def not_multi(procedure_list):
    new_list = []
    for pro in procedure_list:
        new_list.append(pro)
        new_pro = copy.copy(pro)
        you = re.findall(r'\d+\*\d+',pro)
        you_2 = re.findall(r'[a-zA-Z]+\*[0-9]+', pro)
        you_3 = re.findall(r'[0-9]+\*[a-zA-Z]+',pro)
        if you or you_2 or you_3:
            for i in you:
                new_pro = new_pro.replace(i, str(eval(i)))
            for i in you_2:
                shuzi = re.findall(r'(?<=\*)\d+',i)[0]
                zimu = re.findall(r'[a-zA-Z]+(?=\*)',i)[0]
                new_pro = new_pro.replace(i, shuzi+zimu)
            for i in you_3:
                new_pro = new_pro.replace(i, i.replace('*',''))                
            new_list.append(new_pro)
    return new_list

def calculate_similarity_B012(procedure_list, stu_answer_row):
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
        list_procedure = extract_operator_and_shuzizimu(procedure)
        list_ans = extract_operator_and_shuzizimu(stu_answer_row)
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
def rule_zimusuanshi(stuAns_row, procedure,thres=0.7):
    '''
    stuAns: 
        type: str 
        sample: 'I like food\nyou like what\n'
    procedure_list:
        type: list
        sample: ['Frist try this', 'Then try this']
    '''
    procedure_list = not_multi(huansuan_fuhao(extend_B012([procedure])))
    #procedure_list = extend_overline_pro_list(procedure_list)
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity(procedure_list,row) >=thres:
            decide = 1
    return decide

