#yoloを使用したkeyframeの抽出
import av
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN
import pretrainedmodels as ptm

#yolo用のモジュール
import torch 
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from yolo.util import *
import argparse
import os 
import os.path as osp
from yolo.darknet import Darknet
from yolo.preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import itertools

from model import make_model_color, make_model_type, predict
import copy

def detect_keyframe_by_pyav(file_path):
    ####################################################
    # input->file_path:動画のpath
    # output->key_frames:pyavで抽出したkeyframe
    ####################################################

    # 動画の読み込み
    video_container = av.open(file_path)
    key_frames = []

    # 動画のフレームをループで取得
    for idx, f in enumerate(video_container.decode(video=0)):
        # keyframeであった場合
        if f.key_frame:
            # PIL画像に変換
            img = f.to_image()
            # BGRのnumpy配列に変換
            img = np.array(img,dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            key_frames.append(img)
            # plt.imshow(img)
            # plt.show()

            # img = np.array(img,dtype=np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imshow("",img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    
    return key_frames

def arg_parse():
    ####################################################
    # YOLOのパラメータ
    ####################################################

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "yolo/cfg/yolov3-df2.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolo/yolov3-df2_15000.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

def preprocessing(img, inp_dim):
    ####################################################
    # input->img:入力画像
    #        inp_dim:
    # output->img_:
    #         orig_im:
    #         dim:
    ####################################################
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def detect_cloth_by_yolo(key_frames):
    ####################################################
    # input->key_frames:入力画像のリスト
    # output->detect_cloth_key_frame:yoloで切り抜いた服の画像リスト
    ####################################################
    # 各種変数の定義
    args = arg_parse()
    scales = args.scales
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('yolo/data/df2.names')


    # yoloモデルの定義
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model.eval()

    # 入力画像の前処理
    batches = list(map(preprocessing, key_frames, [inp_dim for x in range(len(key_frames))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)



    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
        
    if batch_size != 1:
        num_batches = len(key_frames) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        

    i = 0
    

    write = False
    #model(get_test_input(inp_dim, CUDA), CUDA)
    
    start_det_loop = time.time()
    
    objs = {}

    # モデル入力
    for batch in im_batches:
        if CUDA:
            batch = batch.cuda()
            
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)

        if type(prediction) == int:
            i += 1
            continue


        prediction[:,0] += i*batch_size
        
    
            
          
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))
            

        i += 1

        
        if CUDA:
            torch.cuda.synchronize()


    # 切り取った衣服のlist返却
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
    colors = pkl.load(open("yolo/pallete", "rb"))

    detect_cloth_key_frames = []
    for out in output:
        #画像の切り抜き
        img = key_frames[int(out[0])]
        c1 = tuple(out[1:3].int().cpu().numpy())
        c2 = tuple(out[3:5].int().cpu().numpy())
        # new_img = img[c1[0]:c2[0], c1[1]:c2[1]]
        new_img = img[c1[1]:c2[1], c1[0]:c2[0]]
        detect_cloth_key_frames.append(new_img)

    torch.cuda.empty_cache()

    return detect_cloth_key_frames



def resize_for_metric(detect_cloth_key_frames):
    ####################################################
    # input->detect_cloth_key_frame:画像のリスト
    # output->output:リサイズした画像のリスト
    ####################################################
    output = []
    for img in detect_cloth_key_frames:
        output.append(cv2.resize(img, (299,299)))

    return output

def change_metric_features(detect_cloth_key_frames, system_parameter):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        detect_cloth_key_frames:画像のリスト
    # output->2種の特徴量を結合したもののリスト
    ####################################################
    # model構築
    model1 = make_model_color(system_parameter)
    model2 = make_model_type(system_parameter)
    # リサイズ
    detect_cloth_key_frames = resize_for_metric(detect_cloth_key_frames)
    # 推論
    color_features = predict(system_parameter, model1, detect_cloth_key_frames)
    type_features = predict(system_parameter, model2, detect_cloth_key_frames)
    return [np.concatenate([x,y]) for (x,y) in zip(color_features, type_features)]

def cloth_dbscan(metric_key_frames, detect_cloth_key_frames):
    ####################################################
    # input->metric_key_frames:特徴量のリスト
    #        detect_cloth_key_frames:画像のリスト
    # output->outputs:DBscanによって選別した服の画像
    #         output_features:返却する画像の特徴量
    ####################################################
    dbscan = DBSCAN(eps=1.0, min_samples=1)
    clusters = dbscan.fit_predict(metric_key_frames)
    
    cluster_list = list(set(clusters))
    outputs = []
    outputs_features = []
    for cluster in cluster_list:
        #同一のクラスを持つもののインデックスをまとめる
        indexs = [i for i, v in enumerate(clusters) if v == cluster]

        #ランダムに一つ返す
        index = random.choice(indexs)
        outputs.append(detect_cloth_key_frames[index])
        outputs_features.append(np.array_split(metric_key_frames[index], 2))
    print(len(outputs))

    return outputs, outputs_features

def bgr2rgb(detect_cloth_key_frames):
    ####################################################
    # input->detect_cloth_key_frames:画像のリスト
    # output->outputs:RGBに変換した画像のリスト
    ####################################################
    outputs = []
    for img in detect_cloth_key_frames:
        outputs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return outputs

def get_keyframe(file_path, system_parameter):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        file_path:動画のpath
    # output->outputs:DBscanによって選別した服の画像
    #         output_features:返却する画像の特徴量
    ####################################################
    # keyframe抽出
    key_frames = detect_keyframe_by_pyav(file_path)

    # Yoloで切り出し
    detect_cloth_key_frames = detect_cloth_by_yolo(key_frames)

    # RGBに変換
    detect_cloth_key_frames = bgr2rgb(detect_cloth_key_frames)

    # Metricで特徴量に変換
    metric_key_frames = change_metric_features(detect_cloth_key_frames, system_parameter)

    # 教師なし学習で単一の衣服の画像list作成
    outputs, output_features = cloth_dbscan(metric_key_frames, detect_cloth_key_frames)

    # for img in outputs:
    #     plt.imshow(img)
    #     plt.show()

    return outputs, output_features

if __name__ == "__main__":
    #file_path = "../../../../videos/Man Texting On The Street.mp4"
    # file_path = "../../../../videos/T-Shirts - 31924.mp4"
    
    # file_path = "../videos/test.mp4"
    # file_path = "../../../../videos/mixkit-man-with-smoke-bomb-on-staircase-413.mp4"

    # root = "/Users/buyuu/Desktop/pystd/git_codes/metric-learning-divide-and-conquer/data/img/my_data/yuichi_kano"
    # import glob
    # file_list2 = glob.glob(root+"/*.mp4")
    # file_list = []
    # for a in file_list2:
    #     if "white" in a and "backwhite" not in a:
    #         file_list.append(a)
    file_list = ["/Users/buyuu/Desktop/pystd/git_codes/metric-learning-divide-and-conquer/data/img/my_data/yuichi_kano/white_smartphone_2.mp4"]

    print(file_list)
    # テスト用にパラメータを取得
    from parameters import set_parameters
    system_parameter = set_parameters()
    for file_path in file_list:
        outputs, output_features = get_keyframe(file_path, system_parameter)
        print(len(outputs))
        for i, out in enumerate(outputs):
            a = cv2.imwrite(file_path.split(".")[0]+"_"+str(i+100)+".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            # print(file_path.split(".")[0]+"_"+str(i)+".jpg")
