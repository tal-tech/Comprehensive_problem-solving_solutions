# coding=utf-8
import os
import ftplib
from tqdm import tqdm
"""
功能:模型上传到ftp服务器,从ftp服务器下载到本地
    (1.使用时注意网络是否ping通 2.axer环境使用ip和公司外网ip不一样)
update by boby 20190719 调整代码结构,将上传和下载两个函数分开
"""

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


def upload_model_itf(ftp):

    local_dir = '/workspace/yangjiabo/project_git/pipeline_for_ikkyyu/Models/ssd' #需要指定--本地目录
    remote_root_dir = '/MODEL/OCR/ssd/V0.6.12' #需要指定--远程地址根目录
    det_rm_name = 'V0.5.08' #需要指定--将要创建的目录(一般为对应版本编号,注意模型的名字和yaml文件也做对应的修改)
    remote_dir=os.path.join(remote_root_dir,det_rm_name)
    try:
        ftp.ftp.cwd(remote_root_dir)
        print('切换至远程目录: %s' % ftp.ftp.pwd())
        ftp.ftp.mkd(det_rm_name)
    except:
        print('检测远程目录已存在 %s' % det_rm_name)

    ftp.UpLoadFileTree(local_dir, remote_dir)


def download_model_itf(ftp):
    local_dir='/workspace/ikkyyu_pipeline/pipeline_for_ikkyyu/Models/ssd'
    remote_dir='/MODEL/OCR/ssd/V0.6.12'
    ftp.DownLoadFileTree(local_dir ,remote_dir)  # 从目标目录下载到本地目录E盘


if __name__ == "__main__":
    host = "10.19.10.63" #axer内网使用该地址
    # host = "221.122.129.3" #外网使用vpn
    userName = "cv"
    passwd = "9vJzscnpc9ZOQ"
    ftp = myFtp(host)
    ftp.Login(userName, passwd)  # 登录，如果匿名登录则用空串代替即可

    # upload_model_itf(ftp)
    download_model_itf(ftp)


    ftp.close()
    print("ok!")
