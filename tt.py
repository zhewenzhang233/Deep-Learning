
import json
import os
import os.path
import sys
from tqdm import tqdm

'''with open('C:/Users/zhewe/Desktop/之江实习资料/generate/data/wiki_zh_2019/wiki_zh/AA/wiki_00', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        print(line)'''

filedir_outer = './data/wiki_zh_2019/wiki_zh/'
filedir_inner_list = os.listdir(filedir_outer)
for i in range(len(filedir_inner_list)):
    tempdir = filedir_outer + filedir_inner_list[i]
    filedir_inner_list[i] = tempdir
print(filedir_inner_list)

final_filepath = filedir_outer + "combined_file.txt"
f = open(final_filepath, "w")

with open(final_filepath, 'w', encoding='UTF-8') as f:
    for i in filedir_inner_list:
        filenames = os.listdir(i)
        for filename in filenames:
            filepath = i + "/" + filename
            for line in open(filepath, encoding='UTF-8'):
                f.writelines(line)
print("finish combine all the files")





