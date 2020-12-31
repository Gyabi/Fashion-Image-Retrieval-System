import cv2
import numpy as np
import json
def get_image_paths(system_parameters, DB=True):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        DB:真偽値でDBかクエリか指定
    # output->image_paths:指定したDB又はクエリのpath
    ####################################################
    root = system_parameters["dataset"]["root"]
    use_DB = system_parameters["dataset"]["use_DB"]
    if DB == True:  
        file_path = system_parameters["dataset"][use_DB]["DB_file_path"]
    else:
        file_path = system_parameters["dataset"][use_DB]["query_file_path"]

    image_paths = []
    with open(file_path) as f:
        for line in f:
            data = line.rstrip().split()
            image_paths.append(root + data[0])
    return image_paths

def read_image_RGB(path, resize_size=299):
    ####################################################
    # input->path:読み込む画像のpath
    #        resize_size:リサイズのサイズ
    # output->im:RGB画像のnumpy配列
    ####################################################
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (resize_size,resize_size))
    return im

def make_input(system_parameters, image_paths):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        image_paths:指定したDB又はクエリのpath
    # output->images:modelに入力する画像のnumpy配列
    ####################################################
    calc = system_parameters["sys"]["calc"]
    images = []
    if calc == True:
        for path in image_paths:
            images.append(read_image_RGB(path))
        images = np.array(images)
    return images

def read_label(system_parameters):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    # output->image_data:評価用ラベルのjsonデータをdict型に変換したもの
    ####################################################
    use_DB = system_parameters["dataset"]["use_DB"]
    json_open = open(system_parameters["dataset"][use_DB]["label_file_path"])
    image_data = json.load(json_open)
    return image_data
