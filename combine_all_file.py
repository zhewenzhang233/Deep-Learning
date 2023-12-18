import json
import random
import os

filedir = './data/all_type_dataset/'
filedir_inner_list = os.listdir(filedir)
for i in range(len(filedir_inner_list)):
    tempdir = filedir + filedir_inner_list[i]
    filedir_inner_list[i] = tempdir
print(filedir_inner_list)


combine_file_path = './data/all_type_dataset/all_data.txt'

with open(combine_file_path, 'a', encoding='UTF-8') as file:
    for filepath in filedir_inner_list:
        lines = open(filepath, 'r', encoding='UTF-8')
        for line in lines:
            temp_s = line.strip('\n')
            file.writelines(temp_s + '\n')
