#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
from general_model import jieda_general_answer_solution


def main():

    # OCR输出的学生答案，按照行拼接为字符串，\n分隔
    stu_answer = '1+1=2\n答:为2。'
    # 标准答案
    answer_online = '2'

    #make predictions
    answer_prediction, answer_line = jieda_general_answer_solution(stu_answer, answer_online)
    # print('答案正误:{}'.format(answer_prediction))
    # print('答案行:{}'.format(answer_line))


if __name__ == "__main__":
    main()
