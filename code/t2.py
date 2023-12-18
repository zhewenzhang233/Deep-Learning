import json

'''with open('./data/wiki_zh_2019/t1.txt', 'w', encoding='UTF-8') as f:
    with open('C:/Users/zhewe/Desktop/之江实习资料/generate/data/wiki_zh_2019/wiki_zh/AA/wiki_00', 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        for line in lines:
            f.writelines(line)
            f.write('\n')'''

with open('./data/wiki_zh_2019/combine1.txt', 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    print('len lines: ', len(lines))
    sum = 0
    for i in range(500000):
        # s = lines[i]
        # s = s.replace('\\n', '')
        sum += len(lines[i])
        # print(lines[i])
    avg = sum/500000
    print("avg length: ", avg)

'''with open('./data/wiki_zh_2019/t1.txt', 'w', encoding='UTF-8') as f:
    with open('./data/wiki_zh_2019/combine.txt', 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        f.writelines(lines[0])
        print("done")'''



