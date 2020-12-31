# GUI改善版（画像でyolo実行可能）
from tkinter import *
import tkinter.ttk as ttk
import tkinter.filedialog
import os
from PIL import Image, ImageTk
from key_frame import get_keyframe
from key_frame import detect_cloth_by_yolo
import numpy as np
import cv2

class Tab1(ttk.Frame):
    def __init__(self,mode, master=None, new_parameter=None, width=None, height=None):
        super().__init__(master=master, width=width, height=height)
        # サイズを維持するための関数
        self.grid_propagate(0)
        self.pack()
        # 最終的に返却するパラメータ
        self.n_para = new_parameter
        self.mode = mode

        # 画像の表示領域
        self.canvas1 = Canvas(self,bg="black", width=100, height=100)
        self.canvas1.grid(row=0, column=0)
        self.canvas1.photo = None
        self.image_on_canvas = self.canvas1.create_image(  # キャンバス上にイメージを配置
            0,  # x座標
            0,  # y座標
            image=self.canvas1.photo,  # 配置するイメージオブジェクトを指定
            tag="illust",  # タグで引数を追加する。
            anchor=NW  # 配置の起点となる位置を左上隅に指定
        )

        self.button1 = Button(self, text="Open Explorer", command=self.open_explorer1)
        self.button1.grid(row=1, column=0)


    def open_explorer1(self):
        # fTyp = [("jpg","*.jpg"),("mp4","*.mp4;")]
        fTyp = [("","*")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)

        self.n_para.parameter["mode"] = self.mode
        self.n_para.parameter[self.mode]["query1"]["path"] = file
        self.set_image(file)

    def set_image(self, file):
        if file.split(".")[-1] == "jpg":
            img = Image.open(open(file, "rb"))
            img.thumbnail((100,100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
        else:
            self.n_para.parameter[self.mode]["query1"]["video"] = True
            img = Image.open(open("動画アイコン.jpg", "rb"))
            img.thumbnail((100,100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            
        self.canvas1.photo = img
        self.canvas1.itemconfig(self.image_on_canvas, image=self.canvas1.photo)


class Tab4(ttk.Frame):
    def __init__(self,mode, master=None, new_parameter=None, width=None, height=None):
        super().__init__(master=master, width=width, height=height)
        self.grid_propagate(0)
        self.pack()
        self.n_para = new_parameter
        self.mode = mode

        self.label1 = Label(self, text="色画像")
        self.label1.grid(row=0,column=0)
        self.label2 = Label(self, text="形状画像")
        self.label2.grid(row=0,column=1)

        self.canvas1 = Canvas(self,bg="black", width=100, height=100)
        self.canvas1.grid(row=1, column=0)
        self.canvas1.photo = None
        self.image_on_canvas1 = self.canvas1.create_image(0,0,image=self.canvas1.photo,tag="illust",anchor=NW)

        self.canvas2 = Canvas(self,bg="black", width=100, height=100)
        self.canvas2.grid(row=1, column=1)
        self.canvas2.photo = None
        self.image_on_canvas2 = self.canvas2.create_image(0,0,image=self.canvas2.photo,tag="illust",anchor=NW)

        self.button1 = Button(self, text="Open Explorer", command=self.open_explorer1)
        self.button1.grid(row=2, column=0)
        self.button2 = Button(self, text="Open Explorer", command=self.open_explorer2)
        self.button2.grid(row=2, column=1)

    def open_explorer1(self):
        # fTyp = [("jpg","*.jpg"),("mp4","*.mp4")]
        fTyp = [("","*")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        self.n_para.parameter["mode"] = self.mode
        self.n_para.parameter[self.mode]["query1"]["path"] = file
        self.set_image1(file)

    def open_explorer2(self):
        fTyp = [("","*")]
        iDir = os.path.abspath(os.path.dirname(__file__))
        file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
        self.n_para.parameter["mode"] = self.mode
        self.n_para.parameter[self.mode]["query2"]["path"] = file
        self.set_image2(file)

    def set_image1(self, file):
        if file.split(".")[-1] == "jpg":
            img = Image.open(open(file, "rb"))
            img.thumbnail((100,100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
        else:
            self.n_para.parameter[self.mode]["query1"]["video"] = True
            img = Image.open(open("動画アイコン.jpg", "rb"))
            img.thumbnail((100,100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
        self.canvas1.photo = img
        self.canvas1.itemconfig(self.image_on_canvas1, image=self.canvas1.photo)

    def set_image2(self, file):
        if file.split(".")[-1] == "jpg":
            img = Image.open(open(file, "rb"))
            img.thumbnail((100,100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
        else:
            self.n_para.parameter[self.mode]["query2"]["video"] = True
            img = Image.open(open("動画アイコン.jpg", "rb"))
            img.thumbnail((100,100), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
        self.canvas2.photo = img
        self.canvas2.itemconfig(self.image_on_canvas2, image=self.canvas2.photo)


class NotebookSample(ttk.Frame):
    def __init__(self, master, new_parameter):
        super().__init__(master)
        self.create_widgets(new_parameter)
        self.pack()

    def create_widgets(self, new_parameter):
        note = ttk.Notebook(self)
        note.pack()
        note1 = Tab1("normal",note,new_parameter,width=300,height=200)
        note2 = Tab1("color",note,new_parameter,width=300,height=200)
        note3 = Tab1("type",note,new_parameter,width=300,height=200)
        note4 = Tab4("concat",note,new_parameter,width=300,height=200)
        note.add(note1,text="通常")
        note.add(note2,text="色")
        note.add(note3,text="形状")
        note.add(note4,text="混合")

class New_parameter():
    def __init__(self):
        self.parameter = {
            "mode":None,
            "normal":{
                "query1":{
                    "video":False,
                    "path":None,
                    "feature":None
                }
            },
            "color":{
                "query1":{
                    "video":False,
                    "path":None,
                    "feature":None
                }
            },
            "type":{
                "query1":{
                    "video":False,
                    "path":None,
                    "feature":None
                }
            },
            "concat":{
                "query1":{
                    "video":False,
                    "path":None,
                    "feature":None
                },
                "query2":{
                    "video":False,
                    "path":None,
                    "feature":None
                }
            }
        }

#######################################################################################
# ここまで通常の設定画面
#######################################################################################

class Key_select(ttk.Frame):
    def __init__(self, master=None, keyframes=None):
        super().__init__(master=master)
        self.index = None
        self.var = IntVar()
        self.var.set(0)
        for i, image in enumerate(keyframes):   
            # ラベル
            label = Label(self, text=str(i))
            label.grid(row=int(i/5), column=i%5)

            # 画像
            canvas = Canvas(self,bg="black", width=100, height=100)
            canvas.grid(row=int(i/5)+1, column=i%5)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2pil(image)
            image.thumbnail((100,100), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)
            canvas.photo = image
            canvas.create_image(0,0,image=canvas.photo,tag="illust",anchor=NW)
            # ボタン
            radio = Radiobutton(self, value=i, variable=self.var)
            radio.grid(row=int(i/5)+2, column=i%5)
    def get_index(self):
        return self.var.get()

class Image_select(ttk.Frame):
    def __init__(self, master=None, keyframes=None):
        super().__init__(master=master)
        self.index = None
        self.var = IntVar()
        self.var.set(0)
        for i, image in enumerate(keyframes):   
            # ラベル
            label = Label(self, text=str(i))
            label.grid(row=int(i/5), column=i%5)

            # 画像
            canvas = Canvas(self,bg="black", width=100, height=100)
            canvas.grid(row=int(i/5)+1, column=i%5)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2pil(image)
            image.thumbnail((100,100), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)
            canvas.photo = image
            canvas.create_image(0,0,image=canvas.photo,tag="illust",anchor=NW)
            # ボタン
            radio = Radiobutton(self, value=i, variable=self.var)
            radio.grid(row=int(i/5)+2, column=i%5)

    def get_index(self):
        return self.var.get()
        
def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  # 透過
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def select_key_frame(new_parameter, query_num, system_parameter):
    # pathの取り出し
    input_video_path = new_parameter.parameter[new_parameter.parameter["mode"]][query_num]["path"]
    # yoloに入力
    keyframes, key_features = get_keyframe(input_video_path, system_parameter)
    # 出力画像群をguiで選択
    if len(keyframes) == 1:
        img = keyframes[0]
        feature = key_features[0]
    else:
        root = Tk()
        # root.geometry("400x400")
        x = Key_select(root, keyframes)
        x.pack()
        Button(root, text="実行", command=root.destroy).pack()
        root.mainloop()

        img = keyframes[x.get_index()]
        feature = key_features[x.get_index()]

    # 選択した画像を保存
    os.makedirs("key_query", exist_ok=True)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2pil(img)
    img.save("key_query/"+query_num+".jpg")
    # パラメータの更新
    new_parameter.parameter[new_parameter.parameter["mode"]][query_num]["path"] = "key_query/"+query_num+".jpg"
    new_parameter.parameter[new_parameter.parameter["mode"]][query_num]["feature"] = feature

    return new_parameter

def yolo_for_image(new_parameter, query_num, system_parameter):
    # pathの取り出し
    input_image_path = new_parameter.parameter[new_parameter.parameter["mode"]][query_num]["path"]
    # 画像の読み込み
    images = []
    images.append(cv2.imread(input_image_path))
    # yoloに入力
    detected_images = detect_cloth_by_yolo(images)

    # 出力画像群をguiで選択
    if len(detected_images) == 1:
        img = detected_images[0]
        feature = detected_images[0]
    else:
        root = Tk()
        # root.geometry("400x400")
        x = Image_select(root, detected_images)
        x.pack()
        Button(root, text="実行", command=root.destroy).pack()
        root.mainloop()

        img = detected_images[x.get_index()]

    # 選択した画像を保存
    os.makedirs("key_query", exist_ok=True)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2pil(img)
    img.save("key_query/"+query_num+".jpg")
    # パラメータの更新
    new_parameter.parameter[new_parameter.parameter["mode"]][query_num]["path"] = "key_query/"+query_num+".jpg"

    return new_parameter


def select_image_and_video(system_parameter):
    new_parameter = New_parameter()
    master = Tk()
    master.title("検索設定")
    master.geometry("400x400")
    NotebookSample(master, new_parameter)
    Button(master, text="実行", command=master.destroy).pack()
    master.mainloop()

    if system_parameter["gui"]["use_yolo"] == True:
        # yoloによる画像の指定を行う
        if new_parameter.parameter["mode"] != "concat":
            #混合検索
            if new_parameter.parameter[new_parameter.parameter["mode"]]["query1"]["video"] == True:
                new_parameter = select_key_frame(new_parameter, "query1", system_parameter)
            else:
                # 画像に対するyolo処理
                new_parameter = yolo_for_image(new_parameter, "query1", system_parameter)
        else:
            if new_parameter.parameter[new_parameter.parameter["mode"]]["query1"]["video"] == True:
                new_parameter = select_key_frame(new_parameter, "query1", system_parameter)
            else:
                # 画像に対するyolo処理
                new_parameter = yolo_for_image(new_parameter, "query1", system_parameter)
            if new_parameter.parameter[new_parameter.parameter["mode"]]["query2"]["video"] == True:
                new_parameter = select_key_frame(new_parameter, "query2", system_parameter)
            else:
                # 画像に対するyolo処理
                new_parameter = yolo_for_image(new_parameter, "query2", system_parameter)
    else:
        # 動画を使用している時
        if new_parameter.parameter["mode"] != "concat":
            if new_parameter.parameter[new_parameter.parameter["mode"]]["query1"]["video"] == True:
                new_parameter = select_key_frame(new_parameter, "query1", system_parameter)
        else:
            if new_parameter.parameter[new_parameter.parameter["mode"]]["query1"]["video"] == True:
                new_parameter = select_key_frame(new_parameter, "query1", system_parameter)
            if new_parameter.parameter[new_parameter.parameter["mode"]]["query2"]["video"] == True:
                new_parameter = select_key_frame(new_parameter, "query2", system_parameter)

    return new_parameter.parameter

if __name__ == '__main__':
    from parameters import set_parameters
    system_parameter = set_parameters()
    a = select_image_and_video(system_parameter)

    print(a)