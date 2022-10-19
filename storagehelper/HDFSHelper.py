import os
import sys
import shutil


class HDFSHelper:
    
    def __init__(self, sc, local_fs, local_user='hadoop', hdfs_user='hadoop'):
        self.tag = self.__class__.__name__
        self.local_fs = local_fs
        self.fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
        self.path = sc._jvm.org.apache.hadoop.fs.Path

        self.user_home = os.path.join(os.sep, 'home', local_user) # '/home/hadoop'
        self.hdfs_home = os.path.join(os.sep, 'user', hdfs_user) # '/user/hadoop'


    '''
    remove fs instance
    '''
    def close(self):
        if self.fs is not None:
            print("[{}]: close hdfs".format(self.tag))
            self.fs.close()

    '''
    params:
        username = 'gradients'
    result:
        ['/user/hadoop/gradients/g0.npz', ...]
    '''
    def getFileList(self, dirname = 'gradients'):
        f = os.path.join(self.hdfs_home, dirname)
        
        if self.fs.exists(self.path(f)) == False:
            printf("[Error]: not exist {}".format(f))
            return
        paths = []
        try:
            files = self.fs.listStatus(self.path(f)) #f+os.sep+file.getPath().getName()
            paths = [f+os.sep+file.getPath().getName() for file in files if file.isFile() == True]
        except (FileNotFoundError, OSError) as err:
            print(err)

        return paths

    '''
    params:
        foldername = 'gradients'
    
    '''
    def removeDirectory(self, foldername='gradients'):
        dir_path = self.path(os.path.join(self.hdfs_home, foldername))
        
        try:
            if self.fs.exists(dir_path) == False:
                print("[{}] {}: not exist".format(self.tag, dir_path))
                return False
            filestatus = self.fs.getFileStatus(dir_path) #FileNotFoundError
            if filestatus.isDirectory() == False:
                return False
            self.fs.delete(dir_path, True)
        except (OSError, FileNotFoundError, BaseException, Exception) as err:
            print("[{}] {}".format(self.tag, err))
            return False
        
        print("[{}] {}: removed".format(self.tag, dir_path))
        return True
    
    #If folder already exist delete it and again create it
    '''
    param:
        dirname = 'gradients'
    '''
    def createNewDirectory(self, dir_name='gradients'):
        dir_path = self.path(os.path.join(self.hdfs_home, dir_name))
        
        if self.fs.exists(dir_path) == True:
            print("[{}] {}: remove existing folder".format(self.tag, dir_path))
            self.removeDirectory(dir_name);
        try:
            self.fs.mkdirs(dir_path)
        except OSError as err:
            print("[{}] {}".format(self.tag, err))
            return
        print("[{}] {}: created".format(self.tag, dir_path))

    
    '''
    srd_dir = 'gradients_worker_id1'
    dst_dir = 'gradients'
    file_name = 'gradients_worker_id0.npz',
    '''
    def copyFileFromLocal(self, src_dir, dst_dir, filename):
        src = os.path.join(self.user_home, src_dir, filename)
        dst = os.path.join(self.hdfs_home, dst_dir)
        
        print("[{}] src: {}".format(self.tag, src))
        print("[{}] dst: {}".format(self.tag, dst))
        
        if os.path.exists(src) == False:
            print("[{}] {} src not found.".format(self.tag, src))
            return False
        
        #remove each hdfs files checksum file .crc from local fs
        #removeFileChecksumLocal(username['local'], foldername['local']+worker_id, filename)
        
##        if self.fs.exists(self.path(dst + filename)) == True:
##            print("[{}] {} not found".format(self.tag, dst))
##            return False
        try:
            self.fs.copyFromLocalFile(False, True,self.path(src),self.path(dst))
        except OSError as err:
            print("[{}] {}".format(self.tag, err))
            return False
            
        print("[{}] {} Saved Successfull".format(self.tag, src))
        return True

    '''
    src_dirname = 'gradients' #hdfs dir
    dst_dirname = 'gradients_worker_id5' # local dir
    '''
    def copyFolderToLocal(self, src_dir, dst_dir):
        src = os.path.join(self.hdfs_home,src_dir)
        dst = os.path.join(self.user_home)
        
        if self.fs.exists(self.path(src)) == False:
            print("[{}]: {} not exist".format(self.tag, src))
            return False
        
        print(src)
        print(dest)
        
        self.fs.copyToLocalFile(False, self.path(src), self.path(dest), True)
        #rename local dir 'gradients' --> 'gradients_worker_id0'
        self.local_fs.copyLocalDirectory(src_dir, dst_dir)
        return True

    

    '''
    params:
        src_dir = 'gradients' # hdfs dir
        dst_dir = 'gradients_worker_id5' #local dir
    desc: copy a all files from hdfs src directory to local dst_dir directory
    server --> local_temp --> local
    gradients --> gradients --> gradients_workers_id5
    /user/hadoop/gradients --> /home/hadoop/gradients --> /home/hadoop/gradients_worker_id5
    '''
    def copyFolderToLocal2(self, src_dir, dst_dir):
        src = os.path.join(self.hdfs_home,src_dir) # /user/hadoop/gradients
        dst = os.path.join(self.user_home, src_dir) # /home/hadoop/gradients
        
        if self.fs.exists(self.path(src)) == False:
            print("[{}] {} not exist".format(self.tag, src))
            return
        
        self.local_fs.createLocalDirectory(src_dir)
        
        print("[{}]: copy {} to {}".format(self.tag, src, dst))
        
        file_paths = self.getFileList(src_dir) # ger remote location file list
        for fpath in file_paths:
            print(fpath)
            #copy file '/user/hadoop/gradients/gradients0.npz --> /home/hadoop/gradients/gradients0.npz'
            self.fs.copyToLocalFile(False, self.path(fpath), self.path(dst), True)
        
        #copy local dir '/home/hadoop/gradients' --> /home/hadoop/gradients_worker_id0
        self.local_fs.copyLocalDirectory(src_dir, dst_dir)

        
    # remove hdfs local file checksum file(.crc) for successful upload
    def removeFileChecksumLocal(self, foldername, filename):
        file_path = os.path.join(self.user_home, foldername,filename)
        
        if os.path.exists(file_path) == False:
            print("[{}]: {}: not found".format(self.tag, file_path))
            return True
        
        checksum_path = os.path.join(self.user_home, foldername, '.'+filename+'.crc')
        
        if os.path.exists(checksum_path) == False:
            print("[{}]: No checksum exist for {}".format(self.tag, checksum_path))
            return True
        
        try:
            if os.path.isdir(checksum_path) == True:
                shutil.rmtree(checksum_path)
            else:
                os.remove(checksum_path)
        
        except OSError as err:
            print("[{}]: {}".format(self.tag, err))
            return False
        
        print("[{}]: {} checksum removed".format(self.tag, checksum_path))
