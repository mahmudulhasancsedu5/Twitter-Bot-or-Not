import os
import sys
import shutil

class LocalFSHelper:
    
    def __init__(self, user_name):
        self.tag = self.__class__.__name__
        self.user_root = 'home'
        self.user_name = user_name
        self.user_home = os.path.join(os.sep, self.user_root, self.user_name)
        
    def getLocalDirFileList(self, folder_name):
        dir_path = os.path.join(os.sep, self.user_home, folder_name)

        if os.path.exists(dir_path) == False:
            return []
        
        file_paths = []
        
        for fname in os.listdir(dir_path):
            if fname.endswith(".npz"):
                fpath = dir_path + os.sep + fname;
                file_paths.append(fpath)
                
        return file_paths
    '''
    src_dirname = 'gradients' # local dir
    dst_dirname = 'gradients_worker_id5' # local dir
    '''
    def copyLocalDirectory(self, src_dirname, dst_dirname):
        src = os.path.join(self.user_home, src_dirname)
        dst = os.path.join(self.user_home, dst_dirname)
        
        if os.path.exists(src) == False:
            print("[{}] {} not exist".format(self.tag, src))
            return False

        #if dst directory exist then remove it first before
        #adding new dst directory
        try:
            if os.path.exists(dst) == True:
                shutil.rmtree(dst)
        except (FileNotFoundError, shutil.Error) as err:
            print("[{}]: {}".format(self.tag, err))
            return False
            
        try:
            shutil.copytree(src, dst)
        except (shutil.Error, FileNotFoundError) as err:
            print("[{}]: {}".format(self.tag, err))
            return False
        
        print("[{}]: {} to {} copy successfull".format(self.tag, src, dst))
        return True


    '''
    params:
        dir_name = 'gradients'
    desc: remove directory 'dir_name' from local user home directory
        
    '''
    def removeLocalDirectory(self, dir_name):
        path = os.path.join(self.user_home, dir_name)
        if os.path.exists(path) == False:
            print("[{}]: {}: dir not exist".format(self.tag, path))
            return True
        
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
        except (FileExistsError, FileNotFoundError, PermissionError) as err:
            print("[{}]: {}".format(self.tag, err))
            return False
        
        return True

    
    '''
    params:
        dir_name = 'gradients'
    desc: create a directory in local fs in user home directory
    '''
    def createLocalDirectory(self, dir_name):
        path = os.path.join(self.user_home, dir_name)
        
        if os.path.exists(path) == True:
            self.removeLocalDirectory(dir_name)
        
        try:
            os.mkdir(path)
        except (FileNotFoundError, FileExistsError, Exception) as err:
            print("[{}]: {}".format(self.tag, err))
            return False

        return True
