#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *

def extend_B014(procedure_list):
    new_list = []
    procedure_list = [clean(i) for i in procedure_list]
    for pro in procedure_list:
        new_list.append(pro)
        fanshi = re.findall(r'[\u4E00-\u9FA5]+\s*\+\s*[\u4E00-\u9FA5]+\s*\=\s*[\u4E00-\u9FA5]+',pro)
        left_equal = pro[:pro.find('=')]
        right_equal = pro[pro.find('=')+1:]
        right_zhong = re.findall(r'[\u4E00-\u9FA5]',right_equal)
        left_zhong = re.findall(r'[\u4E00-\u9FA5]+',left_equal)
        if left_zhong[0] == left_zhong[1]:
            new_list.append(left_zhong[0].replace('数','')+'+'+left_zhong[0].replace('数','')+'='+right_zhong[0].replace('数',''))
            new_list.append('两个'+left_zhong[0]+'为'+right_zhong[0])
            new_list.append('2个'+left_zhong[0]+'为'+right_zhong[0])
            new_list.append(left_zhong[0]+'相加为'+right_zhong[0])
            new_list.append(left_zhong[0]+'相加是'+right_zhong[0])
            new_list.append(left_zhong[0]+'相加为'+right_zhong[0].replace('数',''))
            new_list.append(left_zhong[0]+'相加是'+right_zhong[0].replace('数',''))
            new_list.append(left_zhong[0]+'和是'+right_zhong[0])
            new_list.append(left_zhong[0]+'和是'+right_zhong[0].replace('数',''))
            new_list.append(left_zhong[0]+'和为'+right_zhong[0])
            new_list.append(left_zhong[0]+'和为'+right_zhong[0].replace('数',''))
            new_list.append(left_equal.replace('+','加'))
            new_list.append(left_zhong[0].replace('数','')+'加'+left_zhong[0].replace('数','')+'等于'+right_zhong[0])
            new_list.append(left_zhong[0]+'的和为'+right_zhong[0].replace('数',''))
            new_list.append(left_zhong[0]+'的和是'+right_zhong[0].replace('数',''))
    return new_list

def rule_zhongwenjiazhongwen(stuAns_row, procedure,thres):
    '''
    stuAns: 
        type: str 
        sample: 'I like food\nyou like what\n'
    procedure_list:
        type: list
        sample: ['Frist try this', 'Then try this']
    '''
    procedure_list = extend_B014([procedure])
    decide = 0
    row = clean(stuAns_row)
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        tmp_res = [calculate_chinese_similarity(pro, row) for pro in procedure_list]
        if max(tmp_res) >=thres:
            decide = 1
    return decide




