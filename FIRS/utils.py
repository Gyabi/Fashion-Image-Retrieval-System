import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from dataset import read_image_RGB
def save_csv(system_parameters, features, DB=True, f_type="color"):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        features:特徴量
    #        DB:真偽値(DBかクエリか区別)
    #        f_type:色または形状特徴量の区別
    # output->none(csv保存)
    ####################################################
    if DB == True:
        name = "DB"
    else:
        name = "query"

    os.makedirs("features/"+name, exist_ok=True)
    with open("features/"+name+"/"+system_parameters["dataset"]["use_DB"]+"_"+f_type+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        for data in features:
            writer.writerow(data)

def read_csv(system_parameters, DB=True, f_type="color"):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        DB:真偽値(DBかクエリか区別)
    #        f_type:色または形状特徴量の区別
    # output->特徴量ベクトル
    ####################################################
    if DB == True:
        name = "DB"
    else:
        name = "query"
    path = "features/"+name+"/"+system_parameters["dataset"]["use_DB"]+"_"+f_type+".csv"

    return np.loadtxt(path, delimiter=',',dtype="float32")

def map_calculation(system_parameters, search_ids, distances, i, image_data, DB_image_paths, query_image_paths):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        search_ids:検索結果(index)
    #        distances:検索結果(距離)
    #        i:ループ数(クエリが何番目か)
    #        image_data:評価用jsonファイル
    #        DB_image_paths:DBのpathのリスト
    #        query_image_paths:クエリのpathのリスト
    # output->ap:平均適合率(10位)
    ####################################################
    if system_parameters["sys"]["custom"] == True:
        color_index = system_parameters["sys"]["custom_pair"][i][0]
        type_index = system_parameters["sys"]["custom_pair"][i][1]
    else:
        color_index = i
        type_index = i
    #image_pathsについているroot部分を削除する
    root = system_parameters["dataset"]["root"]
    DB_image_paths = [x.replace(root, "") for x in DB_image_paths]
    query_image_paths = [x.replace(root, "") for x in query_image_paths]
    #検索結果を読み出していって演算を行う
    correct_sum = 0
    correct_num = 0
    map_type = system_parameters["sys"]["map_type"]
    for n, search_id in enumerate(search_ids[:10]):
        if map_type == 0:
            if image_data[DB_image_paths[search_id]]["color"] == image_data[query_image_paths[color_index]]["color"]:
                correct_num += 1
                correct_sum += correct_num / (n+1)

        if map_type == 1:
            print(image_data[DB_image_paths[search_id]]["cloth_type"]+":::"+image_data[query_image_paths[type_index]]["cloth_type"])
            if (image_data[DB_image_paths[search_id]]["cloth_type"] == image_data[query_image_paths[type_index]]["cloth_type"]) and (image_data[DB_image_paths[search_id]]["gender"] == image_data[query_image_paths[type_index]]["gender"]):
                correct_num += 1
                correct_sum += correct_num / (n+1)

        if map_type == 2:
            print(image_data[DB_image_paths[search_id]]["color"]+":::"+image_data[query_image_paths[color_index]]["color"])
            print(image_data[DB_image_paths[search_id]]["cloth_type"]+":::"+image_data[query_image_paths[type_index]]["cloth_type"])
            if (image_data[DB_image_paths[search_id]]["color"] == image_data[query_image_paths[color_index]]["color"]) and (image_data[DB_image_paths[search_id]]["cloth_type"] == image_data[query_image_paths[type_index]]["cloth_type"]) and (image_data[DB_image_paths[search_id]]["gender"] == image_data[query_image_paths[type_index]]["gender"]):
                correct_num += 1
                correct_sum += correct_num / (n+1)
    if correct_num == 0:
        ap = 0
    else:
        ap = correct_sum/correct_num

    return ap

def plot_image(search_ids, DB_image_paths, query_image_path, i,system_parameters):
    ####################################################
    # input->system_parameters:システム設定用パラメータ
    #        search_ids:検索結果(index)
    #        i:ループ数(クエリが何番目か)
    #        DB_image_paths:DBのpathのリスト
    #        query_image_paths:クエリのpathのリスト
    # output->none(画像の表示)
    ####################################################
    first_plot = system_parameters["image_out"]["first_plot"]
    show = system_parameters["image_out"]["show"]
    root = system_parameters["dataset"]["root"]
    os.makedirs("result", exist_ok=True)


    cols = 5
    if system_parameters["image_out"]["first_plot"] == True:
        plot_num = len(search_ids) + 1
    else:
        plot_num = len(search_ids)

    if divmod(plot_num,5)[1] == 0:
        rows = plot_num/5
    else:
        rows = plot_num/5 + 1
    axes = []
    fig = plt.figure(figsize=(8,8))

    if rows*cols < plot_num:
        print("please set larger nember than search_ids")
        exit()
    
    if first_plot:
        save_num = i
        axes.append(fig.add_subplot(rows,cols,1))
        sub_title = ("query")
        axes[-1].set_title(sub_title)
        plt.imshow(read_image_RGB(query_image_path, resize_size=399))
        plt.axis("off")
        for i, search_id in enumerate(search_ids):
            axes.append(fig.add_subplot(rows,cols,i+2))
            sub_title = ("result"+str(i+1))
            axes[-1].set_title(sub_title)
            plt.imshow(read_image_RGB(DB_image_paths[search_id], resize_size=399))
            plt.axis("off")

        #出力保存
        fig.tight_layout()
        # plt.subplots_adjust(top=0.9,bottom=0.4)
        plt.savefig("result/"+str(save_num)+".jpg")
        if show:
            plt.show()

    else:
        save_num = i
        for i, search_id in enumerate(search_ids):
            axes.append(fig.add_subplot(rows,cols,i+1))
            sub_title = ("result"+str(i))
            axes[-1].set_title(sub_title)
            plt.imshow(read_image_RGB(DB_image_paths[search_id], resize_size=399))
            # plt.axis("off")

        #出力保存
        fig.tight_layout()
        # plt.subplots_adjust(wspace=0,hspace=0)
        plt.subplots_adjust(top=0.9,bottom=0.4)
        plt.savefig("result/"+str(save_num)+".jpg")
        if show:
            plt.show()

    # if first_plot:
    #     save_num = i
    #     plt.subplot(641),plt.imshow(read_image_RGB(query_image_path, resize_size=600))
    #     for i, search_id in enumerate(search_ids):
    #     # for i, search_id in enumerate(search_ids[:11]):
    #         plt.subplot(6,4,i+2),plt.imshow(read_image_RGB(DB_image_paths[search_id], resize_size=600))
    #         plt.axis("off")
    #     #出力保存
    #     plt.savefig("result/"+str(save_num)+".jpg")
    #     if show:
    #         plt.show()
    # else:
    #     save_num = i
    #     for i, search_id in enumerate(search_ids):
    #         plt.subplot(6,4,i+1),plt.imshow(read_image_RGB(DB_image_paths[search_id], resize_size=399))
    #         plt.axis("off")

    #     #出力保存
    #     plt.savefig("result/"+str(save_num)+".jpg")
    #     if show:
    #         plt.show()