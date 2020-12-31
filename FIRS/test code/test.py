#タイル上に画像表示
import cv2
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import json
import random

def tile_show(pm, pn ,files):
    # 空の入れ物（リスト）を準備
    d = []

    # ファイル名が数字の場合、natsortedで
    # 自然順（ファイル番号の小さい順）に1枚づつ読み込まれる
    for i in files:
        img = Image.open(i)
        img = np.asarray(img)
        #img = cv2.resize(img, (300, 300), cv2.INTER_LANCZOS4)
        d.append(img)

    # タイル状に画像を一覧表示
    fig, ax = plt.subplots(pm, pn, figsize=(10, 10))
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(pm):
        for j in range(pn):
            ax[i, j].xaxis.set_major_locator(plt.NullLocator())
            ax[i, j].yaxis.set_major_locator(plt.NullLocator())
            ax[i, j].imshow(d[pn*i+j], cmap="bone")

    plt.show()


# タイル状に pm × pm 枚配置
pm = 2
pn = 5

# path一覧
root = "\\Users/buyuu/Desktop/pystd/git_codes/metric-learning-divide-and-conquer/data/img/"
db1 = "evaluation_dataset/evaluation_DB2.txt"
db2 = "evaluation_dataset/evaluation_DB_mydata.txt"
q1 = "evaluation_dataset/evaluation_query2.txt"
json_1 = "evaluation_dataset/image_data2.json"

# db1読み込み
files_db1 = []
with open(db1) as f:
    for line in f:
            data = line.rstrip().split()
            files_db1.append(data[0])

# db1のimagedata読み込み
json_open = open(json_1)
image_data1 = json.load(json_open)

# q1読み込み
files_q1 = []
with open(q1) as f:
    for line in f:
            data = line.rstrip().split()
            files_q1.append(data[0])

# db2読み込み
files_db2 = []
with open(db2) as f:
    for line in f:
            data = line.rstrip().split()
            files_db2.append(data[0])

# DB1の内容で2パターン
out1 = []
out2 = []
for path in files_db1:
    if image_data1[path]["color"] == "White" and image_data1[path]["gender"] == "MEN" and image_data1[path]["cloth_type"] == "Shirts_Polos":
        out1.append(root + path)
    if image_data1[path]["color"] == "Red" and image_data1[path]["gender"] == "WOMEN" and image_data1[path]["cloth_type"] == "Dresses":
        out2.append(root + path)

# tile_show(pm,pn,out1)
# クエリ1の例
out3 = []
for path in files_q1[:10]:
    out3.append(root+path)

# tile_show(pm,pn,out3)


# DB2の追加画像例

out4 = []
for path in random.sample(files_db2[:32], 10):
    out4.append(root+path)

tile_show(pm,pn,out4)
# シーン1例
# シーン2例
# 8方向例