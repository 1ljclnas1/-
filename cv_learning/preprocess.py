import csv
import json
import pandas as pd


def json_to_csv(json_path, csv_path, img_root_path):
    # file format conversion
    data = {}
    with open(json_path, 'r', encoding='UTF-8') as f:
        file = json.load(f)
        file = pd.DataFrame(file)
        for index in file.index:
            img_info = file.loc[index]
            img_name_1, img_name_2 = img_info['image_name'].split('/')
            img_name = img_name_1 + '_' + img_name_2
            img_path = img_root_path + img_name + '.png'
            data[index] = {'img_path': img_path,
                           'skin_color': img_info['skin_color'],
                           'lip_color': img_info['lip_color'],
                           'eye_color': img_info['eye_color'],
                           'hair': img_info['hair'],
                           'hair_color': img_info['hair_color'],
                           'gender': img_info['gender'],
                           'earring': img_info['earring'],
                           'smile': img_info['smile'],
                           'frontal_face': img_info['frontal_face'],
                           'style': img_info['style']}
    with open(csv_path, "w", newline='') as f:
        csv_file = csv.writer(f)
        csv_file.writerow(['index', 'img_path', 'skin_color', 'lip_color', 'eye_color', 'hair', 'hair_color', 'gender',
                           'earring', 'smile', 'frontal_face', 'style'])
        for index, img_info in data.items():
            csv_file.writerow([index,
                               img_info['img_path'],
                               img_info['skin_color'],
                               img_info['lip_color'],
                               img_info['eye_color'],
                               img_info['hair'],
                               img_info['hair_color'],
                               img_info['gender'],
                               img_info['earring'],
                               img_info['smile'],
                               img_info['frontal_face'],
                               img_info['style']])


train_json_path = '../cv_learning/FS2K/anno_train.json'
test_json_path = '../cv_learning/FS2K/anno_test.json'
train_csv_path = '../cv_learning/data/train.csv'
test_csv_path = '../cv_learning/data/test.csv'
'''
sketch img path
train_img_path = '../cv_learning/FS2K/train/sketch/'
test_img_path = '../cv_learning/FS2K/test/sketch/'
'''
train_img_path = '../cv_learning/FS2K/train/photo/'
test_img_path = '../cv_learning/FS2K/test/photo/'

json_to_csv(train_json_path, train_csv_path, train_img_path)
json_to_csv(test_json_path, test_csv_path, test_img_path)

