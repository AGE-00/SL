import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

# 歩行者用信号機かチェックし、歩行者用信号機であればその画像を返す
def ReturnPedestrianTrafficLight(results):

    # 結果が0でないことを確認する(ただし、resultsに格納されているのはtraffic lightのみ)
    if len(results.crop()) == 0:
        return None

    # 何かしらの信号機が認識されている場合に以下のコードが実行される
    for traffic_light in results.crop():    # 全ての画像に適用
        img = traffic_light['im']
        img_shape = img.shape
        
        # 縦長画像であれば良い
        if img_shape[0] > img_shape[1]:
            return img    # 歩行者用信号機だったら、その部分を切り抜いた画像を返す
        else:
            return None   # 歩行者用信号機ではなかったらNoneを返す

# 上下から色判定領域を抽出する関数
def extractRedBlueArea(img):
    img_shape = img.shape
    w_c = int(img_shape[1] / 2)
    s = int(img_shape[1] / 6)
    # 上（赤色領域）のエリアにおける抽出画像の中心点を設定
    upper_h_c = int(img_shape[0] / 4)
    # 下（青色領域）のエリアにおける抽出画像の中心点を設定
    lower_h_c = int(img_shape[0] * 3 / 4)

    return [img[upper_h_c - s:upper_h_c + s, w_c - s:w_c+s, :], img[lower_h_c - s:lower_h_c + s, w_c - s:w_c+s, :]]

# 上側の画像と下側の画像をリストにしたものを引数として渡す
def ReturnTrafficLightSignal(img_list):
    
    # 画像データはBGRとする
    upper_img = img_list[0]
    lower_img = img_list[1]

    # 上側（赤）のランプの状態を検出
    upper_red_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()   # 赤色成分の平均値
    upper_blue_nums = cv2.cvtColor(upper_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()  # 青色成分の平均値
    # 差分を求める
    upper_delta = abs(upper_red_nums - upper_blue_nums)

    # 下側（青）のランプの状態を検出
    lower_red_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,0].mean()   # 赤色成分の平均値
    lower_blue_nums = cv2.cvtColor(lower_img, cv2.COLOR_BGR2RGB)[:,:,2].mean()  # 青色成分の平均値
    # 差分を求める
    lower_delta = abs(lower_red_nums - lower_blue_nums)

    if upper_delta >= lower_delta:
        return 'red'
    else:
        return 'blue'

model = torch.hub.load('ultralytics/yolov11', 'yolov11n')
model.classes = [9]

def TrafficLightSignal(imgArray_or_imgPath):
    results = model(imgArray_or_imgPath)
    img = ReturnPedestrianTrafficLight(results)

    if type(img) == np.ndarray:  # 歩行者用信号機の画像があるか確かめる
        RedBlueImgs = extractRedBlueArea(img)
        return ReturnTrafficLightSignal(RedBlueImgs)
    else:
        return None