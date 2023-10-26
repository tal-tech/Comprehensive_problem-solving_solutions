#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from difflib import SequenceMatcher

def extract_integer(text):
    return re.findall(r'-?\d+', text)

def extend_seq_procedure(procedure):
    idx = 1
    new_list = []
    new_strs = []
    for n in procedure.split(','):
        new_strs.append(str(idx)+':'+str(n))
        new_list.append(str(idx)+':'+str(n))
        idx += 1
    new_list.append(','.join(new_strs))
    return list(set([procedure] + new_list))

def delete_xuhao(s):
    str_list1 = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩']
    for str_ in str_list1:
        s = s.replace(str_, '')
    return s
    
def get_idx(len_list, idx):
    l = 0
    for i in range(len(len_list)):
        l += len_list[i]
        if idx < l:
            return i
    return -1

        
def rule_putong_shulie(stuAns, procedure, thres=7):
    ans_list = re.split(r'\n|\\n', stuAns)
    pred_procedure = [0]*len(ans_list)
    stuAns = delete_xuhao(stuAns)
    new_list = extend_seq_procedure(procedure)
    ans_nums = [extract_integer(x) for x in ans_list]
    pro_nums = [extract_integer(x) for x in new_list]
    ans_strs = [''.join(x) for x in ans_nums]
    pro_strs = [''.join(x) for x in pro_nums]
    ans_str = ''.join(ans_strs)
    ans_idx = [len(x) for x in ans_strs]
            
    for pro_str in pro_strs:
        matches = SequenceMatcher(None, ans_str, pro_str).get_matching_blocks()
        if len(matches) == 1:
            continue
        for match in matches[:-1]:
            if match[2] >= thres:
                for i in range(get_idx(ans_idx, match[0]), get_idx(ans_idx, match[0]+match[2]-1)+1):
                    pred_procedure[i] = 1             
    return pred_procedure
