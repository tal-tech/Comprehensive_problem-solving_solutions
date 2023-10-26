#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import re
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

def clean(x):
#     x = "".join(x.split())
    x = str(x)
    x = x.replace('\quad',',')
    x = x.replace('\div','/')
    x = x.replace('\rightarrow','->')
    x = x.replace('\Rightarrow','->')
    x = x.replace('Rightarrow','->')
    x = x.replace('\geq','>=')
    x = x.replace('\leq','<=')
    x = x.replace('\neq','!=')
    x = x.replace('\operatorname', '')
    x = x.replace('\left(','(')
    x = x.replace('\left[','[')
    x = x.replace('\right)',')')
    x = x.replace('\right]',']')
    x = x.replace('\times','*')
    x = x.replace('\equiv','=')
    x = x.replace('\bmod','mod')
    x = x.replace('\mod','mod')
    x = x.replace('\geqslant','>=')
    x = x.replace('\cdots','...')
    x = x.replace('\cdot','.')
    x = x.replace('－', '-')
    x = x.replace('＝', '=')
    x = x.replace('＋', '+')
    x = x.replace('（', '(')
    x = x.replace('）', ')')
    x = x.replace('÷', '/')
    x = x.replace('×', '*')
    x = x.replace('、',',')
    x = x.replace('﹣', '-')
    x = x.replace('\\quad',',')
    x = x.replace('\\div','/')
    x = x.replace('\\rightarrow','->')
    x = x.replace('\\Rightarrow','->')
    x = x.replace('\\geq','>=')
    x = x.replace('\\leq','<=')
    x = x.replace('\\neq','!=')
    x = x.replace('\\operatorname', '')
    x = x.replace('\\left(','(')
    x = x.replace('\\left[','[')
    x = x.replace('\\right]',']')
    x = x.replace('\\right)',')')
    x = x.replace('\\times','*')
    x = x.replace('\\equiv','=')
    x = x.replace('\\bmod','mod')
    x = x.replace('\\mod','mod')
    x = x.replace('\\geqslant','>=')
    x = x.replace('\\cdots','...')
    x = x.replace('\\cdot','.')
    x = x.replace(' ','')
    return x
def replace_xuhao(x, rep):
    xuhao = ['①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩',
             '(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)',
            '（1）','（2）','（3）','（4）','（5）','（6）','（7）','（8）','（9）','（10）']
    for xu in xuhao:
        if xu+':' in x:
            x = x.replace(xu+':', rep)
        elif xu+'：' in x:
            x = x.replace(xu+'：', rep)
        else:
            x = x.replace(xu, rep)
    return x

def calculate_similarity(procedure_list, stu_answer_row):
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
        list_procedure = extract_operator_and_number(procedure)
        list_ans = extract_operator_and_number(stu_answer_row)
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
def calculate_chinese_similarity(procedure,stu_answer_row):
    list_procedure = [x for x in procedure]
    list_ans = [z for z in stu_answer_row]
    intersection = len(set(list_procedure).intersection(list_ans))
    union = len(set(list_procedure)) + len(set(list_ans)) - intersection
    return intersection/union

def extract_float_number(text):
    '''
    text:
        type: str
        sample: '1+1=2'
    '''
    return re.findall(r"\d+\.?\d*",text)
def extract_number(text):
    return re.findall(r'-?\d+(?:\.\d+)?', text)
def extract_operator_and_number(s):
    return re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', s) + re.findall(r'\d+(?:\.\d+)?', s)
def extract_integer(s):
    return re.findall(r'-?\d+', s)
def extract_operator_and_integer(s):
    return re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', s)+re.findall(r'-?\d+', s)
def extract_chi_number(s):
    return re.findall(r'[\u4E00-\u9FA5]',s) + re.findall(r'\d+',s) + re.findall(r'\\div',s)
def extract_operator_and_shuzizimu(s):
    return re.findall(r'\+|-|\*|/|=|!=|>=|<=|>|<|\(|\)|\[|\]', s) + re.findall(r'\d+[a-zA-Z]*|\d*[a-zA-Z]', s)

def get_metrics(test_pred, test_label):
    # print(len(test_pred), len(test_label))
    # print('prediction', test_pred)
    # print('label', test_label)
    precision = precision_score(test_label, test_pred)
    recall = recall_score(test_label,test_pred)
    f1 = f1_score(test_label, test_pred)
    accuracy = accuracy_score(test_label, test_pred)
    print('acc:{}, precision:{}, recall:{}, f1:{}'.format(accuracy,precision,recall,f1))
    return accuracy, precision, recall,f1

def zhuazhun_img(preds,labels,ratio_thres):
    if len(preds) != len(labels):
        raise ValueError('prediction length is not equal to label length')
    sum_this = 0
    for i in range(len(preds)):
        pred = preds[i]
        label = labels[i]
        equal_count = 0
        if len(pred) != len(label):
            raise ValueError('prediction row length is not equal to label row length')
        for j in range(len(pred)):
            equal_count += 1 if pred[j] == label[j] else 0
        ratio = equal_count / len(pred)
        if ratio >= ratio_thres:
            sum_this +=1
    return sum_this/len(preds)

def zhuazhun(test_df,ratio_thres):
    img_list = test_df.image_id.unique()
    sum_this = 0
    for img in img_list:
        test_img = test_df[test_df['image_id']==img]
        ratio = test_img[test_img['label_procedure']==test_img['common']].shape[0]/test_img.shape[0]
        if ratio >= ratio_thres:
            sum_this +=1
    return sum_this/len(img_list)

def get_metrics_have_procedure(df, label_col, pred_col):
    TP = df[(df[label_col]==1)&(df[pred_col]==1)].shape[0]
    FP = df[(df[label_col]==0)&(df[pred_col]==1)].shape[0]
    TN = df[(df[label_col]==0)&(df[pred_col]==0)].shape[0]
    FN = df[(df[label_col]==1)&(df[pred_col]==0)].shape[0]
    precision = 0 if (TP + FP)==0 else TP / (TP + FP)
    recall = 0 if (TP + FN)==0 else TP / (TP + FN)
    f1 = 0 if (precision + recall)==0 else 2 * (precision * recall) / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    # print('正例','precision {}, recall {}, f1 {}, acc {}'.format(precision,recall,f1,acc))
    neg_precision = 0 if (TN + FN)==0 else TN / (TN + FN)
    neg_recall = 0 if (TN + FP)==0 else TN / (TN + FP)
    neg_f1 = 0 if (neg_precision + neg_recall)==0 else 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
    # print('负例','precision {}, recall {}, f1 {}'.format(neg_precision,neg_recall,neg_f1))
    return acc,precision,recall,f1


