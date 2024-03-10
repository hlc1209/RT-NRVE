# %%
import os
import numpy as np
from shutil import copyfile

ORIG_PATH = "./crm/"
TARG_PATH = "./small/train/crm/"

file_list = [f for f in os.listdir(ORIG_PATH) if ("npy" in f) and (not "p232" in f) and (not "p257" in f)]

p = np.random.RandomState(seed=0).permutation(len(file_list))
file_list = list(np.array(file_list)[p])

file_list = file_list[:20000]

if not os.path.isdir(TARG_PATH):
    os.mkdir(TARG_PATH)

for file_name in file_list:
    copyfile(ORIG_PATH + file_name, TARG_PATH + file_name)