#!pip install split-folders

import numpy as np
import pandas as pd
import os
import keras
import matplotlib.pyplot as plt
import shutil
import splitfolders

os.mkdir('/kaggle/working/main_directory')

kaggle_input_path = '/kaggle/input/covid19-radiography-database/COVID-19_Radiography_Dataset/'

#Copy files to one folder
shutil.copytree(kaggle_input_path + 'Normal/images','/kaggle/working/main_directory/Normal')
shutil.copytree(kaggle_input_path+ 'COVID/images','/kaggle/working/main_directory/COVID')

#split into train/val/test splits
splitfolders.ratio('/kaggle/working/main_directory/', '/kaggle/working/output',seed=1337,ratio=(0.8,0.1,0.1),move=True)

