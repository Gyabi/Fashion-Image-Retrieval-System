# 提案システム実験用実行コード
import json
import parameters
from dataset import *
from model import *
from utils import *
from search import *
def main():
    print("画像データ読み込み開始")
    # 実験に使用するパラメータを読み込み
    system_parameters = parameters.set_parameters()
    # DB画像PATH取得
    DB_image_paths = get_image_paths(system_parameters, DB=True)
    # DB画像読み込み
    DB_images = make_input(system_parameters, DB_image_paths)
    # query画像PATH取得
    query_image_path = get_image_paths(system_parameters, DB=False)
    # query画像読み込み
    query_images = make_input(system_parameters, query_image_path)
    # 評価用のラベル読み込み
    image_data = read_label(system_parameters)
    print("画像データ読み込み終了")

    print("model構築開始")
    # モデル構築
    if "color" in system_parameters["sys"]["use_feature"]:
        model_color = make_model_color(system_parameters)
    if "type" in system_parameters["sys"]["use_feature"]:
        model_type = make_model_type(system_parameters)
    print("model構築終了")


    # 特徴量抽出及び保存
    print("特徴量抽出開始")
    DB_features = []
    query_features = []
    if system_parameters["sys"]["calc"] == True:
        if "color" in system_parameters["sys"]["use_feature"]:
            c_DB_features = predict(system_parameters, model_color, DB_images)
            DB_features.append(c_DB_features)
            c_query_features = predict(system_parameters, model_color, query_images)
            query_features.append(c_query_features)

            # 保存
            save_csv(system_parameters, c_DB_features, DB=True, f_type="color")
            save_csv(system_parameters, c_query_features, DB=False, f_type="color")
        if "type" in system_parameters["sys"]["use_feature"]:
            t_DB_features = predict(system_parameters, model_type, DB_images)
            DB_features.append(t_DB_features)
            t_query_features = predict(system_parameters, model_type, query_images)
            query_features.append(t_query_features)

            # 保存
            save_csv(system_parameters, t_DB_features, DB=True, f_type="type")
            save_csv(system_parameters, t_query_features, DB=False, f_type="type")
    else:
        if "color" in system_parameters["sys"]["use_feature"]:
            c_DB_features = read_csv(system_parameters, DB=True, f_type="color")
            DB_features.append(c_DB_features)
            c_query_features = read_csv(system_parameters, DB=False, f_type="color")
            query_features.append(c_query_features)
        if "type" in system_parameters["sys"]["use_feature"]:
            t_DB_features = read_csv(system_parameters, DB=True, f_type="type")
            DB_features.append(t_DB_features)
            t_query_features = read_csv(system_parameters, DB=False, f_type="type")
            query_features.append(t_query_features)
    print("特徴量抽出終了")

    print("近似最近傍探索開始")
    # 近似最近傍探索
    if system_parameters["sys"]["dimension_reduction"] == False:
        results = search(system_parameters, DB_features, query_features)
    else:
        results = search2(system_parameters, DB_features, query_features)
    print("近似最近傍探索終了")

    print("精度計算開始")
    # 精度計算
    output = []
    for i, result in enumerate(results):
        search_ids = result[0]
        distances = result[1]
        map_ = map_calculation(system_parameters, search_ids, distances, i, image_data, DB_image_paths, query_image_path)
        output.append(map_)

        plot_image(search_ids, DB_image_paths, query_image_path[i], i,system_parameters)
    print("精度計算終了")

    with open("result/result.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(output)

if __name__ == "__main__":
    main()