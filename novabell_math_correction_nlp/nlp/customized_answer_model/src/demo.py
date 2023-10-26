#! /usr/bin/env python3
# -*- coding: UTF-8 -*-
from answer_model import customization_model, load_configuration

def main():
    # 读取定制库
    config_file_path = 'customization.csv'
    config_dict = load_configuration(config_file_path)

    # 做预测
    que_id = '57f10104cfe54841b3f5ea3718b3b3ff'
    stu_answer = 'z=0:x+2y=100\nz=1\nx+2y=96\nz=2:x+2y=92\nz=3:x+2y=88\nz=25,x+2y=0\n\\frac{100}{2}+1\\quad\\frac{96}{2}+1+\\frac{92}{2}+1+\\frac{88}{2}+1+\\cdots\\cdot+\\frac{0}{2}+1\n=2++50+48+46+44+\\cdots\\cdot\\cdot+0\n6\n6\n=2+\\left(0+50\\right)\\times26\\div2\n=26+650\n=676\\left(组\\right)\n答:有676组'
    # print('学生回答:{}'.format(stu_answer))
    answer_prediction, answer_line = customization_model(config_dict, que_id, stu_answer)
    # print('答案正误:{}'.format(answer_prediction))
    # print('答案行{}'.format(answer_line))


if __name__ == "__main__":
    main()