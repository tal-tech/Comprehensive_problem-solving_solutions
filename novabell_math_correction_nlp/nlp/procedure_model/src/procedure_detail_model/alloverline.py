#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
import collections
from procedure_model.src.procedure_utils import *


# \overline{abc}+\overline{acb}+\overline{bac}+\overline{bca}+\overline{cab}+\overline{cba}
def extend_all_overlines(procedure):     
    new_list = []
    if re.match(r'^(?:\\overline\s*\{[a-zA-Z0-9]+\}[\+,\-\s*、;；，]\s*)+\\overline\s*\{[a-zA-Z0-9]+\}$', procedure):
        matches = re.findall(r'\\overline\s*\{[a-zA-Z0-9]+\}', procedure)
        new_list.append(','.join(matches))
        new_list.append(' '.join(matches))
        new_list.append('、'.join(matches))
        new_list.append('，'.join(matches))

    return list(set([procedure]+new_list))

def delete_xuhao(s):
    str_list1 = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)']
    for str_ in str_list1:
        s = s.replace(str_, '')
    return s

def extract_overlines(pro):    
    pro = delete_xuhao(pro)
    over_list = []
    if re.match(r'^(?:\\overline\s*\{[a-zA-Z0-9]+\}[\+,\-\s*、;；，]\s*)+\\overline\s*\{[a-zA-Z0-9]+\}$', pro):
        over_list = re.findall(r'\\overline\s*\{([a-zA-Z0-9]+)\}', pro)
    return over_list

def rule_all_overline(stuAns_row, procedure, thres):
    row = clean(stuAns_row)
    procedure = clean(procedure)
    procedure_list = extend_all_overlines(procedure)
    decide = 0
    if any(x in row for x in procedure_list):
        decide = 1
    else:
        if calculate_similarity(procedure_list, row) >= thres:
            decide = 1
        else:
            row_overlines = extract_overlines(row)
            for pro in procedure_list:
                pro_overlines = extract_overlines(pro)
                if len(row_overlines) > 0 and len(pro_overlines) > 0 and collections.Counter(row_overlines) == collections.Counter(pro_overlines):
                    decide = 1
                    break
            # if decide == 0:
            #     plus_rule = [extract_operator_and_number(x) for x in procedure_list]
            #     if extract_operator_and_number(row) in plus_rule:
            #         decide = 1
    return decide