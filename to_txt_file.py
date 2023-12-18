import json
from tqdm import tqdm
import os.path

filedir_outer = './data/wiki_zh_2019/wiki_zh/'
filedir_inner_list = os.listdir(filedir_outer)
for i in range(len(filedir_inner_list)):
    tempdir = filedir_outer + filedir_inner_list[i]
    filedir_inner_list[i] = tempdir
print(filedir_inner_list)


def open_json(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            sample = line.split('text": "')
            sample[-1] = sample[-1][0: -3]
            sample_temp = sample[-1].replace('\\n', '')
            with open('./data/wiki_zh_2019/combine1.txt', 'a', encoding='UTF-8') as file:
                file.writelines(sample_temp + '\n')


for i in tqdm(filedir_inner_list):
    filenames = os.listdir(i)
    for filename in filenames:
        filepath = i + "/" + filename
        open_json(filepath)
print("finish combine all the files")

