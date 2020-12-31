import nmslib
import numpy as np

def search(system_parameters, DB_features, query_features):
    ####################################################
    # input->system_parameters:システムのパラメータ
    #          DB_features:DB内の画像をベクトル変換したリスト
    #          query_features:クエリ内の画像をベクトル変換したリスト
    # output->result:result:検索結果のidと距離が格納されたlist
    ####################################################
    result = []
    if system_parameters["sys"]["search_method"] == "concat":
        if system_parameters["sys"]["custom"] == False:
            if len(system_parameters["sys"]["use_feature"]) == 1:
                for i, query_feature in enumerate(query_features[0]):
                    if i == 0:
                        index = nmslib.init(method="hnsw", space="l2")
                        index.addDataPointBatch(DB_features[0])
                        index.createIndex({"post":2},print_progress=True)
                    
                    search_ids, distances = index.knnQuery(query_feature, k=system_parameters["sys"]["search_num"])

                    result.append([])
                    result[i].append(search_ids)
                    result[i].append(distances)

            elif len(system_parameters["sys"]["use_feature"]) == 2:
                for i, (x,y) in enumerate(zip(query_features[0], query_features[1])):
                    if i == 0:
                        index = nmslib.init(method="hnsw", space="l2")
                        index.addDataPointBatch([np.concatenate([x_,y_]) for (x_,y_) in zip(DB_features[0], DB_features[1])])
                        index.createIndex({"post":2},print_progress=True)

                    search_ids, distances = index.knnQuery(np.concatenate([x,y]), k=system_parameters["sys"]["search_num"])

                    result.append([])
                    result[i].append(search_ids)
                    result[i].append(distances)
        else:#カスタム検索
            if len(system_parameters["sys"]["use_feature"]) == 2:
                for i, pair in enumerate(system_parameters["sys"]["custom_pair"]):
                    if i == 0:
                        index = nmslib.init(method="hnsw", space="l2")
                        index.addDataPointBatch([np.concatenate([x_,y_]) for (x_,y_) in zip(DB_features[0], DB_features[1])])
                        index.createIndex({"post":2},print_progress=True)

                    search_ids, distances = index.knnQuery(np.concatenate([query_features[0][pair[0]],query_features[1][pair[1]]]), k=system_parameters["sys"]["search_num"])

                    result.append([])
                    result[i].append(search_ids)
                    result[i].append(distances)
                    print("-----------------------------------------------------------------------")
                    
            else:
                print("Please set color and type in use_feature")

    return result


def search2(system_parameters, DB_features, query_features):
    ####################################################
    # input->system_parameters:システムのパラメータ
    #          DB_features:DB内の画像をベクトル変換したリスト
    #          query_features:クエリ内の画像をベクトル変換したリスト
    # output->result:result:検索結果のidと距離が格納されたlist
    ####################################################
    # 次元削減結合法
    # 10近傍の平均距離計算
    if system_parameters["sys"]["custom"] == False:
        dis_aves1 = []
        for i, query_feature in enumerate(query_features[0]):
            if i == 0:
                index = nmslib.init(method="hnsw", space="l2")
                index.addDataPointBatch(DB_features[0])
                index.createIndex({"post":2},print_progress=False)
            search_ids, distances = index.knnQuery(query_feature, k=10)
            dis_aves1.append(np.average(distances))

        dis_aves2 = []
        for i, query_feature in enumerate(query_features[1]):
            if i == 0:
                index = nmslib.init(method="hnsw", space="l2")
                index.addDataPointBatch(DB_features[1])
                index.createIndex({"post":2},print_progress=False)
            search_ids, distances = index.knnQuery(query_feature, k=10)
            dis_aves2.append(np.average(distances))
    else:
        dis_aves1 = []
        for i, pair in enumerate(system_parameters["sys"]["custom_pair"]):
            if i == 0:
                index = nmslib.init(method="hnsw", space="l2")
                index.addDataPointBatch(DB_features[0])
                index.createIndex({"post":2},print_progress=False)
            search_ids, distances = index.knnQuery(query_features[0][pair[0]], k=10)
            dis_aves1.append(np.average(distances))

        dis_aves2 = []
        for i, pair in enumerate(system_parameters["sys"]["custom_pair"]):
            if i == 0:
                index = nmslib.init(method="hnsw", space="l2")
                index.addDataPointBatch(DB_features[1])
                index.createIndex({"post":2},print_progress=False)
            search_ids, distances = index.knnQuery(query_features[1][pair[1]], k=10)
            dis_aves2.append(np.average(distances))

    # DBより各次元の標準偏差からポイントを付与
    db_std1 = np.argsort(np.std(DB_features[0], axis=0))
    db_std2 = np.argsort(np.std(DB_features[1], axis=0))
    # pointの最大値を計算
    point_max = 0
    for a in range(len(query_features[0][0])):
        point_max += a
    # 平均距離の比を取得
    ave_ratio = {}
    for i, (dis_ave1, dis_ave2) in enumerate(zip(dis_aves1, dis_aves2)):
        ave_ratio[i] = {}
        if dis_ave2 > dis_ave1:
            # if (dis_ave1/dis_ave2) <= 0.33:
            #     ave_ratio[i]["point"] = point_max *0.7
            # else:
            #     ave_ratio[i]["point"] = point_max
            # ave_ratio[i]["point"] = point_max * 0.75
            ave_ratio[i]["point"] = point_max * point_magnification(dis_ave2, dis_ave1)
            ave_ratio[i]["target"] = 1
        else:
            # if (dis_ave2/dis_ave1) <= 0.33:
            #     ave_ratio[i]["point"] = point_max * 0.7
            # else:
            #     ave_ratio[i]["point"] = point_max
            ave_ratio[i]["point"] = point_max * point_magnification(dis_ave1, dis_ave2)
            # ave_ratio[i]["point"] = point_max * 0.75
            ave_ratio[i]["target"] = 2
    # 各クエリで使用するインデックスを指定
    for i in range(len(dis_aves1)):
        indexs = []
        point_calc = ave_ratio[i]["point"]
        if ave_ratio[i]["target"] == 1:
            # for j in np.sort(np.arange(np.amax(db_std1)+1)):
            for j in np.sort(np.arange(np.amax(db_std1)+1))[::-1]:
                if point_calc - j >= 0:
                    point_calc -= j
                    indexs.append(int(np.where(db_std1 == j)[0]))
                else:
                    break
            indexs.sort()        
            ave_ratio[i]["indexs"]  = indexs

        elif ave_ratio[i]["target"] == 2:
            # for j in np.sort(np.arange(np.amax(db_std2)+1)):
            for j in np.sort(np.arange(np.amax(db_std2)+1))[::-1]:
                if point_calc - j >= 0:
                    point_calc -= j
                    indexs.append(int(np.where(db_std2 == j)[0]))
                else:
                    break
            indexs.sort()        
            ave_ratio[i]["indexs"]  = indexs



    result = []
    if system_parameters["sys"]["search_method"] == "concat":
        if system_parameters["sys"]["custom"] == False:
            if len(system_parameters["sys"]["use_feature"]) == 1:
                for i, query_feature in enumerate(query_features[0]):
                    if i == 0:
                        index = nmslib.init(method="hnsw", space="l2")
                        index.addDataPointBatch(DB_features[0])
                        index.createIndex({"post":2},print_progress=False)
                    
                    search_ids, distances = index.knnQuery(query_feature, k=system_parameters["sys"]["search_num"])

                    result.append([])
                    result[i].append(search_ids)
                    result[i].append(distances)

            elif len(system_parameters["sys"]["use_feature"]) == 2:
                for i, (x,y) in enumerate(zip(query_features[0], query_features[1])):
                    index = nmslib.init(method="hnsw", space="l2")
                    if ave_ratio[i]["target"] == 1:
                        index.addDataPointBatch([np.concatenate([x_[ave_ratio[i]["indexs"]],y_]) for (x_,y_) in zip(DB_features[0], DB_features[1])])
                    elif ave_ratio[i]["target"] == 2:
                        index.addDataPointBatch([np.concatenate([x_,y_[ave_ratio[i]["indexs"]]]) for (x_,y_) in zip(DB_features[0], DB_features[1])])
                    index.createIndex({"post":2},print_progress=False)

                    if ave_ratio[i]["target"] == 1:
                        search_ids, distances = index.knnQuery(np.concatenate([x[ave_ratio[i]["indexs"]],y]), k=system_parameters["sys"]["search_num"])
                    elif ave_ratio[i]["target"] == 2:
                        search_ids, distances = index.knnQuery(np.concatenate([x,y[ave_ratio[i]["indexs"]]]), k=system_parameters["sys"]["search_num"])


                    result.append([])
                    result[i].append(search_ids)
                    result[i].append(distances)
        else:#カスタム検索
            if len(system_parameters["sys"]["use_feature"]) == 2:
                for i, pair in enumerate(system_parameters["sys"]["custom_pair"]):
                    index = nmslib.init(method="hnsw", space="l2")
                    if ave_ratio[i]["target"] == 1:
                        index.addDataPointBatch([np.concatenate([x_[ave_ratio[i]["indexs"]],y_]) for (x_,y_) in zip(DB_features[0], DB_features[1])])
                    elif ave_ratio[i]["target"] == 2:
                        index.addDataPointBatch([np.concatenate([x_,y_[ave_ratio[i]["indexs"]]]) for (x_,y_) in zip(DB_features[0], DB_features[1])])
                    index.createIndex({"post":2},print_progress=False)

                    if ave_ratio[i]["target"] == 1:
                        search_ids, distances = index.knnQuery(np.concatenate([query_features[0][pair[0]][ave_ratio[i]["indexs"]],query_features[1][pair[1]]]), k=system_parameters["sys"]["search_num"])
                    elif ave_ratio[i]["target"] == 2:
                        search_ids, distances = index.knnQuery(np.concatenate([query_features[0][pair[0]],query_features[1][pair[1]][ave_ratio[i]["indexs"]]]), k=system_parameters["sys"]["search_num"])

                    result.append([])
                    result[i].append(search_ids)
                    result[i].append(distances)
                    print("-----------------------------------------------------------------------")
                    
            else:
                print("Please set color and type in use_feature")

    return result


def point_magnification(bigger,smaller):
    alpha = 3
    x = bigger/smaller
    if x >= alpha:
        return 0.8
    else:
        # 線形
        return -0.2/alpha*x + 1
        # 2次_1
        # return -0.2/alpha/alpha*x*x + 1
        # 2次_2
        # return -0.2/(alpha*alpha-2*alpha*alpha*alpha)*x*x + 0.4*alpha/(alpha*alpha-2*alpha*alpha*alpha)*x + 1