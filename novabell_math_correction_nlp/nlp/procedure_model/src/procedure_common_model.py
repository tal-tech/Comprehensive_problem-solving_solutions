#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from procedure_model.src.procedure_utils import *
from procedure_model.src.procedure_classifier import procedure_classify
from procedure_model.src.procedure_detail_model.dengcha import rule_dengcha
from procedure_model.src.procedure_detail_model.gongbeishu import rule_gongbeishu
from procedure_model.src.procedure_detail_model.gongyueshu import rule_gongyueshu
from procedure_model.src.procedure_detail_model.jiefangcheng import rule_jiefangcheng
from procedure_model.src.procedure_detail_model.qita import rule_qita
from procedure_model.src.procedure_detail_model.rotate_number import rule_rotate_number
from procedure_model.src.procedure_detail_model.suanshi import rule_suanshi
from procedure_model.src.procedure_detail_model.yushu import rule_yushu
from procedure_model.src.procedure_detail_model.zhongwen import rule_chinese
from procedure_model.src.procedure_detail_model.shulie import rule_pingfang_shulie
from procedure_model.src.procedure_detail_model.pingfangsuanshi import rule_pingfang_suanshi
from procedure_model.src.procedure_detail_model.overline import rule_overline
from procedure_model.src.procedure_detail_model.shuzi_zhongwen import rule_shuzi_zhongwen
from procedure_model.src.procedure_detail_model.ziweishu import rule_ziweishu
from procedure_model.src.procedure_detail_model.pingfang_budengshi import rule_pingfangbudengshi
from procedure_model.src.procedure_detail_model.zimusuanshi import rule_zimusuanshi
from procedure_model.src.procedure_detail_model.zhongwenjiazhongwen import rule_zhongwenjiazhongwen
from procedure_model.src.procedure_detail_model.putongshulie import rule_putong_shulie
from procedure_model.src.procedure_detail_model.shuzizimu import rule_shuzi_zimu
from procedure_model.src.procedure_detail_model.alloverline import rule_all_overline

thres_dict = {
    '循环小数': 0.53,
    '公约数': 0.53,
    '公倍数': 0.53,
    '除法余数':0.53,
    '解方程':0.8,
    '等差数列':0.53,
    '普通算式':0.7,#以前0.53
    '中文描述':0.8,
    '普通数列':7,
    '平方数列':1,
    '平方算式': 0.7,
    'overline相关': 0.54,
    '数字中文':0.53,
    '字为数':0.53,
    '平方不等式':0.53,
    '字母算式':0.7,
    '中文+中文':0.53,
    '数字跟字母运算':0.76,
    '全部overline':0.25,
    '其他':0.54,
}
rule_dict = {
    '循环小数': rule_rotate_number,
    '公约数': rule_gongyueshu,
    '公倍数': rule_gongbeishu,
    '除法余数':rule_yushu,
    '解方程':rule_qita,
    '等差数列':rule_dengcha,
    '普通算式':rule_suanshi,
    '其他':rule_qita,
    '中文描述':rule_chinese,
    '普通数列': rule_putong_shulie,
    '平方数列':rule_pingfang_shulie,
    '平方算式':rule_pingfang_suanshi,
    '数字中文':rule_shuzi_zhongwen,
    '字为数':rule_ziweishu,
    '平方不等式':rule_pingfangbudengshi,
    '字母算式':rule_zimusuanshi,
    '中文+中文':rule_zhongwenjiazhongwen,
    '数字跟字母运算':rule_shuzi_zimu,
    '全部overline':rule_all_overline,
    'overline相关':rule_overline,
}

def have_procedure(stuAns, answer_list, procedure_list, thres1=7):
    try:
        stuAns = clean(str(stuAns))
        stuAns_new = re.sub(r'\n|\\n','',stuAns)
        answer_list = [clean(x) for x in answer_list]
        procedure_list = [clean(x) for x in procedure_list]
        procedure = ';'.join(procedure_list)
        answer = ';'.join(answer_list)
        rows = [x for x in re.split(r'\n|\\n', stuAns) if len(x)>0]
        stuAns_num = extract_number(replace_xuhao(stuAns,''))
        ans_num = extract_number(answer)
        stuAns_opr_num = extract_operator_and_number(replace_xuhao(stuAns,''))
        ans_opr_num = extract_operator_and_number(answer)
    #     print(rows)
        if stuAns_new in answer:      # 完全匹配答案
            return 0
        if len([x for x in rows if len(re.findall(r'^答[\s\S]*', x))==0 or len(x)>=len(procedure)]) <= 0: # 只有‘答’且字数少于关键过程
            return 0
        if len(rows) <= 1 and len(ans_num)!=0 and len(stuAns_num)<=len(ans_num):   # 只有一行
            return 0
        if len(ans_opr_num)!=0 and len(stuAns_opr_num)!=0 and all(x in ans_opr_num for x in stuAns_opr_num) and len(stuAns_opr_num) <= len(ans_opr_num):
            return 0
        # 答案含有不只一个
        if len(rows) <= len(answer_list) and len(ans_num)!=0 and len(stuAns_num)<=len(ans_num):
            return 0
        if len(re.sub(r'答:|答：|答|,|;|，|；|。|、', '', stuAns_new)) < thres1:
            return 0
        return 1
    except Exception as e:
        raise ValueError('Exception occurs when judge have procedure. Exception={}'.format(e))

def key_procedure(stuAns, procedure_list):
    pred_procedure = []
    matched_procedure = []
    i = 0
    rows = [x.strip() for x in re.split(r'\n|\\n', stuAns) if len(x.strip())>0]
    for row in rows:
        # print(row)
        decide = 0
        this_row_pred = []
        row = row.lower()
        for pro in procedure_list:
            temp_pred = 0
            pro = pro.lower()
            type_this_pro = procedure_classify(pro)
            if type_this_pro == '普通数列':
                temp_pred = rule_dict[type_this_pro](stuAns,pro,thres_dict[type_this_pro])[i]    
            else:
                temp_pred = rule_dict[type_this_pro](row,pro,thres_dict[type_this_pro])
            this_row_pred.append(temp_pred)
            if temp_pred == 1:
                matched_procedure.append(pro)
        if sum(this_row_pred) > 0:
            decide = 1
        pred_procedure.append(decide)
        i += 1
    matched_procedure = list(set(matched_procedure))
    return matched_procedure,pred_procedure,this_row_pred

def procedure_solution(stuAns, answer_list, procedure_list):
    res_json = {"procedure_status": '000',
        "procedure_label": [],
        "procedure_list": []
    }
    have_ = have_procedure(stuAns, answer_list, procedure_list)
    if have_ == 0:
        return res_json
    try:
        matched_list, pred_list, temp = key_procedure(stuAns, procedure_list)
        if len(matched_list) == 0:
            res_json['procedure_status'] = '002'
        else:
            res_json['procedure_status'] = '001'
        res_json['procedure_label'] = pred_list
        res_json['procedure_list'] = matched_list
        return res_json
    except Exception as e:
        raise ValueError('Exception occurs when catching key procedures. Exception={}'.format(e))


'''
def main(stuAns, procedure_list):
    pred_procedure = []
    for row in stuAns.split('\n'):
        decide = 0
        this_row_pred = []
        for pro in procedure_list:
            type_this_pro = procedure_classify(pro)
            this_row_pred.append(rule_dict[type_this_pro](row,pro,thres_dict[type_this_pro]))
        if sum(this_row_pred) >0:
            decide = 1
        pred_procedure.append(decide)
    return pred_procedure,this_row_pred

def main_2(stuAns, procedure_list):
    pred_procedure = []
    for row in stuAns.split('\\*n')[:-1]:
        decide = 0
        this_row_pred = []
        for pro in procedure_list:
            type_this_pro = procedure_classify(pro)
            this_row_pred.append(rule_dict[type_this_pro](row,pro,thres_dict[type_this_pro]))
        if sum(this_row_pred) >0:
            decide = 1
        pred_procedure.append(decide)
    return pred_procedure,this_row_pred
'''


if __name__ == "__main__":
    
    stuAns = '解:设他折了n个千纸鹤\nn\div3余2\nn\div4余3\nn\div7余6\n补数相同都为1\nn+1=3,4,7的公倍数\n\left[3,4,7\right]=84,168,252\cdots\cdots\n84-1=83'
    answer_list = [83]
    procedure_list = ['[3,4,7]-1']
    print(procedure_solution(stuAns, answer_list, procedure_list))
    
    
