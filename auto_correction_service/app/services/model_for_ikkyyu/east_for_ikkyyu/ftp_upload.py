# coding=utf-8
import os
import ftplib
from tqdm import tqdm


class myFtp:
    ftp = ftplib.FTP()
    bIsDir = False
    path = ""

    def __init__(self, host, port=21):
        # self.ftp.set_debuglevel(2) #打开调试级别2，显示详细信息
        # self.ftp.set_pasv(0)      #0主动模式 1 #被动模式
        self.ftp.connect(host, port)

    def Login(self, user, passwd):
        self.ftp.login(user, passwd)
        print(self.ftp.welcome)

    def DownLoadFile(self, LocalFile, RemoteFile):  # 下载当个文件
        file_handler = open(LocalFile, 'wb')
        print(file_handler)
        self.ftp.retrbinary("RETR %s" % (RemoteFile), file_handler.write)  # 接收服务器上文件并写入本地文件
        file_handler.close()
        return True

    def UpLoadFile(self, LocalFile, RemoteFile):
        if os.path.isfile(LocalFile) == False:
            return False
        file_handler = open(LocalFile, "rb")
        self.ftp.storbinary('STOR %s' % RemoteFile, file_handler, 4096)  # 上传文件
        file_handler.close()
        return True

    def UpLoadFileTree(self, LocalDir, RemoteDir):
        if os.path.isdir(LocalDir) == False:
            return False
        print("LocalDir:", LocalDir)
        LocalNames = os.listdir(LocalDir)
        print("list:", LocalNames)
        print(RemoteDir)
        self.ftp.cwd(RemoteDir)
        for Local in tqdm(LocalNames):
            src = os.path.join(LocalDir, Local)
            if os.path.isdir(src):
                self.UpLoadFileTree(src, Local)
            else:
                self.UpLoadFile(src, Local)
        self.ftp.cwd("..")
        return

    def DownLoadFileTree(self, LocalDir, RemoteDir):  # 下载整个目录下的文件
        print("remoteDir:", RemoteDir)
        if os.path.isdir(LocalDir) == False:
            os.makedirs(LocalDir)
        self.ftp.cwd(RemoteDir)
        RemoteNames = self.ftp.nlst()
        print("RemoteNames", RemoteNames)
        print(self.ftp.nlst("/del1"))
        for file in RemoteNames:
            Local = os.path.join(LocalDir, file)
            if self.isDir(file):
                self.DownLoadFileTree(Local, file)
            else:
                self.DownLoadFile(Local, file)
        self.ftp.cwd("..")
        return

    def show(self, list):
        result = list.lower().split(" ")
        if self.path in result and "<dir>" in result:
            self.bIsDir = True

    def isDir(self, path):
        self.bIsDir = False
        self.path = path
        # this ues callback function ,that will change bIsDir value
        self.ftp.retrlines('LIST', self.show)
        return self.bIsDir

    def close(self):
        self.ftp.quit()


if __name__ == "__main__":
    host = "10.19.10.63"
    userName = "cv"
    passwd = "9vJzscnpc9ZOQ"


    detection_remote_dir='/MODEL/OCR/east/'
    recognition_remote_dir='/MODEL/OCR/crnn/'


    det_rm_name = 'V0.6.19'
    reg_rm_name = 'V0.6.18'

    ftp = myFtp(host)
    ftp.Login(userName, passwd)  # 登录，如果匿名登录则用空串代替即可

    try:
        ftp.ftp.cwd(detection_remote_dir)
        print('切换至远程目录: %s' % ftp.ftp.pwd())
        ftp.ftp.mkd(det_rm_name)
    except:
        print('检测远程目录已存在 %s' % det_rm_name)

    try:
        ftp.ftp.cwd(recognition_remote_dir)
        print('切换至远程目录: %s' % ftp.ftp.pwd())
        ftp.ftp.mkd(reg_rm_name)
    except:
        print('识别远程目录已存在 %s' % reg_rm_name)


    det_remote_dir=os.path.join(detection_remote_dir,det_rm_name)
    reg_remote_dir=os.path.join(recognition_remote_dir,reg_rm_name)

    srcDir = '/workspace/boby/model/EAST_tf/ftp/V0.6.19'
    # ftp.DownLoadFileTree( '/workspace/yangjiabo/project_git/pipeline_for_ikkyyu/Models/ssd','/MODEL/OCR/ssd/V0.6.12/')  # 从目标目录下载到本地目录E盘
    ftp.UpLoadFileTree(srcDir, det_remote_dir)

    # ftp.DownLoadFile('E:/study/r2101-ROOT-20170428.zip','/owt/20170504/r2101-ROOT-20170428.zip')
    # ftp.UpLoadFile('E:/study/bak.txt','/owt/20170504/bak.txt')
    ftp.close()
    print("ok!")
