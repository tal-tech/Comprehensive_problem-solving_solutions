#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import sys
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, ROOT_PATH)

import pandas as pd
import re
import json
from customized_answer_model.src.utils import locateAnswerLine
from customized_answer_model.src import solution

# 加载定制库字段，返回字典用于批改
def load_configuration(config_file_path):
    config = pd.read_csv(config_file_path)
    config_dict = {}
    for i in range(config.shape[0]):
        que_id = config['question_id'].iloc[i]
        solution_name = config['solution_name'].iloc[i]
        correct_answer = eval(config['correct_answer'].iloc[i])
        wrong_answer = config['wrong_answer'].iloc[i]
        wrong_answer = None if type(wrong_answer)==float else eval(wrong_answer)
        keywords = config['keywords'].iloc[i]
        keywords = None if type(keywords)==float else eval(keywords)

        config_dict[que_id] = {'solution_name':solution_name,
                                        'correct_answer' : correct_answer,
                                        'wrong_answer': wrong_answer,
                                        'keywords' : keywords}
    return config_dict

# 答案批改模型总接口
def customization_model(config, que_id, stuAns):
    '''
    Args:
    config: 定制库, type: dict
    que_id: 旧题的que_id， type:str
    stuAns: 学生回答，即OCR输出（需要把OCR按行输出的list合并成string，每一行结尾加上换行符'\n'）， type:str, 

    Returns:
    result: 0, 1, -1， 0错误 1正确 -1定制库内不包含que_id, type:int
    ans_list: 定制库中的答案, type: list
    '''
	# 输入que_id不在定制库覆盖范围内，返回-1和[]
    if(not que_id in config):
        # print('que_id {} not in database'.format(que_id))
        return -1, []

    solution_name = config[que_id]['solution_name']
    method_to_call = getattr(solution, solution_name)

    ans_list = config[que_id]['correct_answer']
    wrong_lst = config[que_id]['wrong_answer']
    keywords = config[que_id]['keywords']

	# 定位答案行
    answer_line = locateAnswerLine(stuAns, keywords=keywords)

    result = method_to_call(stuAns, keywords, ans_list, wrong_lst)
    return result, answer_line