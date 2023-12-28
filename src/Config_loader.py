import os
import sys
import fnmatch
import yaml

def find_yaml_file(path,file_name):
    # 遍历指定路径下的所有文件和文件夹
    for root, dirs, files in os.walk(path):
        # 遍历当前文件夹中的所有文件
        for file in files:
            # 判断文件名是否匹配a.yaml
            if fnmatch.fnmatch(file, file_name):
                return os.path.join(root, file)
    # 如果没有找到匹配的文件，返回None
    return None


def load_config(file_name):
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_file=find_yaml_file(_BASE_DIR,file_name)
    with open(config_file, 'r') as f:     # 用with读取文件更好
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs

class Config_loader():
    def __init__(self,file_name):
        _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_file=find_yaml_file(_BASE_DIR,file_name)
        with open(config_file, 'r') as f:     # 用with读取文件更好
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        
    
    def find_yaml_file(self,path,file_name):
        # 遍历指定路径下的所有文件和文件夹
        for root, dirs, files in os.walk(path):
            # 遍历当前文件夹中的所有文件
            for file in files:
                # 判断文件名是否匹配a.yaml
                if fnmatch.fnmatch(file, file_name):
                    return os.path.join(root, file)
        # 如果没有找到匹配的文件，返回None
        return None
    
    def get_params(self):
        return self.configs
    