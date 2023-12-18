import pandas as pd
import os
import json
import random

def get_image_path(image_name):
    return os.path.join(image_name)

def create_dataset(csv_path,right_folder):
    df = pd.read_csv(csv_path)
    
    dataset = []
    
    for _, row in df.iterrows():
        data = {
            "name": row["left"],
            "left_img": get_image_path(row["left"] + ".jpg"),
            "right_img": []row[f"c{i}"] + ".jpg" for i in range(20)
        }
        dataset.append(entry)
        
        '''data["right_img"].append(get_image_path(row["right"] + ".jpg"))
        
        # 获取right文件夹下所有的图片名称，并移除已经被使用的图片名称
        all_right_images = os.listdir(right_folder)
        all_right_images.remove(row["right"] + ".jpg")
        
        # 为right_matrixs添加其它19张图片
        additional_images = random.sample(all_right_images, 19)
        data["right_img"].extend([get_image_path(img) for img in additional_images])
        
        dataset.append(data)'''
        
    return json.dumps(dataset, indent=4)

csv_path = 'test_candidates.csv'
left_folder = './test/left'
right_folder = './test/right'

json_dataset = create_dataset(csv_path, right_folder)

# 输出或保存为json文件
print(json_dataset)
# 或者
with open("test.json", "w") as file:
    file.write(json_dataset)

