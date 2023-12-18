import json

'''f = open('C:/Users/zhewe/Desktop/之江实习资料/generate/data/wiki_zh_2019/wiki_zh/AA/wiki_00', 'r', encoding='UTF-8')

lines = f.readlines()

full_src_lst = []
full_tgt_lst = []
for i, example in enumerate(lines):
    full_src_lst.append(example['id'])
    full_tgt_lst.append(example['title'])

print(full_tgt_lst,full_src_lst)'''

'''with open('C:/Users/zhewe/Desktop/之江实习资料/generate/data/wiki_zh_2019/wiki_zh/AA/wiki_00', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    s = []
    for line in lines:
        temp = line.split(",")
        print(line)
        s.append(temp)
    print(s[0][0])'''

'''f = open('C:/Users/zhewe/Desktop/之江实习资料/generate/data/wiki_zh_2019/wiki_zh/AA/wiki_00', 'r', encoding='UTF-8')

lines = f.readlines()

full_src_lst = []

for i, example in enumerate(lines):
    line = json.loads(example)
    full_src_lst.append(line['text'])

print(full_src_lst[0])'''

'''file_path = "./data/e2e_data/src1_test.txt"
with open(file_path, encoding="utf-8") as f:
    lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                    and len(line.split('||')) == 2)]
src_lines, tgt_lines = list(zip(*lines))
src_lines = list(src_lines)
tgt_lines = list(tgt_lines)
print(src_lines[0])
print(tgt_lines[0])'''

'''f = open('./data/wiki_zh_2019/combine.json', 'r', encoding='UTF-8')

lines = f.readlines()

full_src_lst = []

for i, example in enumerate(lines):
    line = json.loads(example)
    full_src_lst.append(line['text'])


with open('./data/wiki_zh_2019/t1.txt', 'r', encoding='UTF-8') as f:
    for i in range(10):
        temp = full_src_lst[i].replace('\\n', '')
        print(repr(temp))
        # f.writelines(temp + '\n')'''
full_src_lst = []
full_tgt_lst = []

for i, example in enumerate(lines):
    full_src_lst.append(example['dialogue'])
    full_tgt_lst.append(example['summary'])

assert len(full_tgt_lst) == len(full_src_lst)

print('data length : {}'.format(len(full_src_lst)))

print("begin to tokenize {} data ...".format(self.__class__.__name__))
sys.stdout.flush()

# cnt = []
# for src in full_src_lst:
#     cnt.append(len(tokenizer(src, add_special_tokens=False, truncation=True)["input_ids"]))
# print(sum(cnt) / len(cnt))
#
# cnt = []
# for tgt in full_tgt_lst:
#     cnt.append(len(tokenizer(tgt, add_special_tokens=False, truncation=True)["input_ids"]))
# print(sum(cnt) / len(cnt))
#
# exit()

# 156.54452891664403
# 25.50237578061363





