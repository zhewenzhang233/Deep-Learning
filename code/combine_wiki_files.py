import os.path
import os

# 脚本文件，用来合并所有的wiki-zh-2019中的数据
# wiki_zh_2019 数据有两层文件夹，先读取第一层文件夹的名字并把他们加入filedir_inner_list中
filedir_outer = './data/wiki_zh_2019/wiki_zh/'
filedir_inner_list = os.listdir(filedir_outer)
for i in range(len(filedir_inner_list)):
    tempdir = filedir_outer + filedir_inner_list[i]
    filedir_inner_list[i] = tempdir
print(filedir_inner_list)

# 创建新的txt文件
final_filepath = filedir_outer + "combined_file.txt"
f = open(final_filepath, "w")

# 把所有文件写入新的txt文件
with open(final_filepath, 'w', encoding='UTF-8') as f:
    for i in filedir_inner_list:
        filenames = os.listdir(i)
        for filename in filenames:
            filepath = i + "/" + filename
            for line in open(filepath, encoding='UTF-8'):
                f.writelines(line)
            f.write('\n')
print("finish combine all the files")
