# -*- coding:utf-8 -*-

# 功能:从指定路径中下载图片
# boby
# 2017.03.07

import os
import random
import shutil
import tqdm

# 随机抽取文件
def copyFile(fileDir, tarDir, randomCount):
    pathDir = os.listdir(fileDir)
    print("tatal:" + str(len(pathDir)))
    sample = random.sample(pathDir, randomCount)
    # print(sample)
    for name in sample:
        print(name)
        # 复制
        shutil.copyfile(fileDir + name, tarDir + name)
        # 剪切
        # shutil.move(fileDir + name, tarDir + name)

# 随机抽取文件,图片和对应的TXT同时移动
def mvFile(imgDir,txtDir,imgTarDir,txtTarDir,randomCount):

    pathDir = os.listdir(imgDir)
    print("tatal:" + str(len(pathDir)))
    sample = random.sample(pathDir, randomCount)

    for name in sample:
        (file_name,extension)=os.path.splitext(name)
        img_path=os.path.join(imgDir,name)
        txt_path=os.path.join(txtDir,file_name+".txt")
        img_tar_path = os.path.join(imgTarDir, name)
        txt_tar_path = os.path.join(txtTarDir, file_name + ".txt")

        shutil.move(img_path,img_tar_path)
        shutil.move(txt_path,txt_tar_path)


# 打印现在文件夹的文件数
def printCurrentCount(fileDir, tarDir):
    srcCurCount = 0
    tarCurCount = 0
    pathDir = os.listdir(fileDir)
    for file in pathDir:
        if file.endswith('jpg'):
            srcCurCount += 1
    print("srcCurCount:" + str(srcCurCount))

    sDir = os.listdir(tarDir)
    for file in sDir:
        if file.endswith('jpg'):
            tarCurCount += 1
    print("tarCurCount:" + str(tarCurCount))


# 重命名
def rename(path):
    i = 0
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件
        i = i + 1
        Olddir = os.path.join(path, files);  # 原来的文件路径
        print(Olddir)
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
            continue;
        # filename = os.path.splitext(files)[0];  # 文件名
        # filetype = os.path.splitext(files)[1];  # 文件扩展名
        Newdir = os.path.join(path, files.replace("20190306_SmallSupermarket_4000", ""))  # 新的文件路径
        print(Newdir)
        os.rename(Olddir, Newdir)  # 重命名

# 随机抽取文件
def copyAllFile(fileDir, tarDir):
    pathDir = os.listdir(fileDir)
    for name in tqdm.tqdm(pathDir):
        # print(name)
        # 复制
        # shutil.copyfile(fileDir + name, tarDir + name)
        # 剪切
        try:
            shutil.move(fileDir + name, tarDir + name)
        except:
            print("error"+fileDir + name)
# 主函数
if __name__ == '__main__':

    for i in range(8,10):
        fileDir = "/share/bing_data/upload/190515_3352_3673_toc_9k_reg_"+str(i)+"/"
        print(fileDir)
        tarDir = '/share/bing_data/toc_train/190515_3352_3673_toc_9k_reg/'
        copyAllFile(fileDir, tarDir)