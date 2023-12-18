import json
from tqdm import tqdm
import os.path

filedir_outer = './data/wiki_zh_2019/wiki_zh/'
filedir_inner_list = os.listdir(filedir_outer)
for i in range(len(filedir_inner_list)):
    tempdir = filedir_outer + filedir_inner_list[i]
    filedir_inner_list[i] = tempdir
print(filedir_inner_list)

final_filepath = filedir_outer + "combined_file.txt"
f = open(final_filepath, "w")

print("finish combine all the files")


def open_json(filepath):
    with open(filepath, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            example = json.loads(line)
            d = {'text': example['text']}
            with open('./data/wiki_zh_2019/combine.json', 'a', encoding='UTF-8') as file:
                json.dump(d, file, ensure_ascii=False)
                file.write('\n')


for i in tqdm(filedir_inner_list):
    filenames = os.listdir(i)
    for filename in filenames:
        filepath = i + "/" + filename
        open_json(filepath)




'''f = open('./data/wiki_zh_2019/wiki_zh/combined_file.txt', 'r', encoding='UTF-8')

lines = f.readlines()

full_src_lst = []

for i, example in tqdm(enumerate(lines)):
    print(example)
    line = json.loads(example)
    full_src_lst.append(line['text'])

print(full_src_lst)'''
