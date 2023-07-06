import os
import shutil
import random
from torchvision import transforms, datasets, utils
import json
import torch
class Data_Preprocess:
    def __init__(self):
        pass
    def run(self):
        root_file='D:/github/datasets/AlexNet_demo_dataset'
        def mkfile(file):
            if not os.path.exists(file):
                os.makedirs(file)
        flower_class = [cla for cla in os.listdir(root_file) ]
        for cla in flower_class:
            mkfile(root_file+'/train/'+cla)
        for cla in flower_class:
            mkfile(root_file+'/val/'+cla)
        split_rate=0.1
        for cla in flower_class:
            cla_path=root_file+"/"+cla+"/"
            images=os.listdir(cla_path)
            images_num=len(images)
            val_index=random.sample(images,k=int(images_num*split_rate))
            for index,image in enumerate(images):
                if image in val_index:
                    image_path=cla_path+image
                    new_path=root_file+'/val/'+cla+'/'
                    shutil.copy(image_path,new_path)
                else:
                    image_path=cla_path+image
                    new_path=root_file+'/train/'+cla+'/'
                    shutil.copy(image_path,new_path)
                print("\r[{}] processing [{}/{}]".format(cla, index + 1, images_num), end="")
            print()
        print("data processing done!")
