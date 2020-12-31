def set_parameters():
    parameters = {
        "model":{
            "weight":{
                "color":"weights/metric_color_weight.pth",
                "type":"weights/metric_id_weight.pth"
            }
        },
        "sys":{
            "use_feature":["type"],#使用する特徴量"color" or "type"
            "search_num":10,#検索数
            "custom":False,#組み合わせによる検索
            # "custom_pair":[(49,15),(32,47),(5,0),(53,5),(2,5),(10,28),(29,53),(5,30),(47,49),(53,47)],
            "custom_pair":[(49,15),(32,47),(5,0),(53,5),(29,53),(5,30),(47,49),(53,47),
                           (49,43),(5,1),(7,41),(2,14),(29,30),(5,42),(47,41),(53,7),
                           (41,30),(3,5),(30,28),(8,53),(5,30),(57,49),(53,47),
                           (41,15),(3,0),(22,25),(8,30),(59,42),(57,41),(51,7)
                        ],
            "search_method":"concat",#検索手法"concat" or "step"
            "step_search_num":{"0":50, "1":10},#段階検索に使用する検索数
            "map_type":2,#MAPの計算に使用する情報    0：color, 1:type, 2:color&type
            "calc":False,
            "calc_DB":False,#GUI使用時のみ使う
            "calc_query":True,#GUI使用時のみ使う
            "dimension_reduction":False#次元削限による結合を使用するか
        },
        "dataset":{
            "use_DB":"DB2",
            "root":"/Users/buyuu/Desktop/pystd/git_codes/metric-learning-divide-and-conquer/data/img/",
            "DB1":{#600枚DB
                "DB_file_path":"evaluation_dataset/evaluation_DB2.txt",
                "query_file_path":"evaluation_dataset/evaluation_query2.txt",
                "label_file_path":"evaluation_dataset/image_data2.json"
            },
            "DB2":{#自分の写真を使用
                "DB_file_path":"evaluation_dataset/evaluation_DB_mydata.txt",
                "query_file_path":"evaluation_dataset/evaluation_query_mydata.txt",
                "label_file_path":"evaluation_dataset/image_data_mydata.json"
            }
        },
        "image_out":{
            "first_plot":False,
            "show":False
        },
        "gui":{#画像に対してyoloを使用するか
            "use_yolo":False
        }
    }

    return parameters