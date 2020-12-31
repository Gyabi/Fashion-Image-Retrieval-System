# 提案システム実装プログラム
import json
import parameters
from dataset import *
from model import *
from utils import *
from search import *
from gui import select_image_and_video
def update_parameter(system_parameters, new_parameter):
    system_parameters["sys"]["calc"] = True
    if new_parameter["mode"] == "concat":
        system_parameters["sys"]["use_feature"] = ["color", "type"]
        system_parameters["sys"]["custom"] = True
        system_parameters["sys"]["custom_pair"] = [(0,1)]
    elif new_parameter["mode"] == "normal":
        system_parameters["sys"]["custom"] = False
        system_parameters["sys"]["use_feature"] = ["color", "type"]
    elif new_parameter["mode"] == "color":
        system_parameters["sys"]["custom"] = False
        system_parameters["sys"]["use_feature"] = ["color"]
    else:
        system_parameters["sys"]["custom"] = False
        system_parameters["sys"]["use_feature"] = ["type"]


    return system_parameters


def main():
    # 実験に使用するパラメータを読み込み
    system_parameters = parameters.set_parameters()
    # GUIによる選択
    new_parameter = select_image_and_video(system_parameters)
    # パラメータ更新
    system_parameters = update_parameter(system_parameters, new_parameter)
    # DB画像PATH取得
    DB_image_paths = get_image_paths(system_parameters, DB=True)
    # DB画像読み込み
    DB_images = make_input(system_parameters, DB_image_paths)
    # query画像PATH取得
    query_image_path = []
    query_image_path.append(new_parameter[new_parameter["mode"]]["query1"]["path"])
    if new_parameter["mode"] == "concat":
        query_image_path.append(new_parameter[new_parameter["mode"]]["query2"]["path"])
    # query画像読み込み
    query_images = make_input(system_parameters, query_image_path)



    # モデル構築
    if "color" in system_parameters["sys"]["use_feature"]:
        model_color = make_model_color(system_parameters)
    if "type" in system_parameters["sys"]["use_feature"]:
        model_type = make_model_type(system_parameters)


    # 特徴量抽出及び保存
    DB_features = []
    query_features = []
    if system_parameters["sys"]["calc_DB"] == True:
        if "color" in system_parameters["sys"]["use_feature"]:
            c_DB_features = predict(system_parameters, model_color, DB_images)
            DB_features.append(c_DB_features)

            # 保存
            save_csv(system_parameters, c_DB_features, DB=True, f_type="color")
        if "type" in system_parameters["sys"]["use_feature"]:
            t_DB_features = predict(system_parameters, model_type, DB_images)
            query_features.append(t_DB_features)

            # 保存
            save_csv(system_parameters, t_DB_features, DB=True, f_type="type")
    else:
        if "color" in system_parameters["sys"]["use_feature"]:
            c_DB_features = read_csv(system_parameters, DB=True, f_type="color")
            DB_features.append(c_DB_features)

        if "type" in system_parameters["sys"]["use_feature"]:
            t_DB_features = read_csv(system_parameters, DB=True, f_type="type")
            DB_features.append(t_DB_features)



    if system_parameters["sys"]["calc_query"] == True:
        if "color" in system_parameters["sys"]["use_feature"]:
            c_query_features = predict(system_parameters, model_color, query_images)
            query_features.append(c_query_features)

            # 保存
            # save_csv(system_parameters, c_query_features, DB=False, f_type="color")
        if "type" in system_parameters["sys"]["use_feature"]:
            t_query_features = predict(system_parameters, model_type, query_images)
            query_features.append(t_query_features)

            # 保存
            # save_csv(system_parameters, t_query_features, DB=False, f_type="type")
    else:
        if "color" in system_parameters["sys"]["use_feature"]:
            c_query_features = read_csv(system_parameters, DB=False, f_type="color")
            query_features.append(c_query_features)
        if "type" in system_parameters["sys"]["use_feature"]:
            t_query_features = read_csv(system_parameters, DB=False, f_type="type")
            query_features.append(t_query_features)

    # 近似最近傍探索
    if system_parameters["sys"]["dimension_reduction"] == False:
        results = search(system_parameters, DB_features, query_features)
    else:
        results = search2(system_parameters, DB_features, query_features)


    # 精度計算
    output = []
    for i, result in enumerate(results):
        search_ids = result[0]
        distances = result[1]

        plot_image(search_ids, DB_image_paths, query_image_path[i], i,system_parameters)

    print("-----------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()