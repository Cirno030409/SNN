import os
import pathlib
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import MiniBatchDictionaryLearning
from tqdm import tqdm

IMG_DIR = "256_ObjectCategories"  # 学習する画像データのディレクトリ

LOAD_FROM_NPY = False  # 保存された画像リストから読み込む
IMG_LIST_PATH = "values/images.npy"  # 読み込む画像リストのパス

SHOW_LOADED_DICT = False  # 保存された辞書を読み込んで表示する
DICT_PATH = "values/Dictionaries.npy"  # 読み込む辞書のパス


# 画像リスト読み込み
def ImageListFile2Array(img_path_list):
    imgArray = None
    n = 0
    print("loading images...")
    for image_file in tqdm(img_path_list):
        # 画像を読み込み
        im = Image.open(image_file)

        im_resize = im.resize((256, 256))

        im_crop = crop(im_resize, (8, 8))

        # im.show()
        # グレースケール変換
        gray_im = ImageOps.grayscale(im_crop)

        # gray_im.show()

        data = np.asarray(gray_im)
        data = data.reshape(1, data.size)
        if imgArray is None:
            imgArray = data
        else:
            imgArray = np.concatenate((imgArray, data))
    return imgArray


# 画像のパスリストの読み込み
def make_image_path_list(dir_path):
    path_list = []
    for path in pathlib.Path(dir_path).glob("**/*.jpg"):
        path_list.append(str(path))
    return path_list


# 画像をランダムに切り出す
def crop(image, crop_size):
    # 画像サイズ取得
    w, h = image.width, image.height

    ### 画像サイズをランダムに決定 ###
    # 画像の左上の座標(height 0, width 0)
    h0 = np.random.randint(0, h - crop_size[0])
    w0 = np.random.randint(0, w - crop_size[1])

    # 画像の右下の座標(height 1, width 1)
    h1 = h0 + crop_size[0]
    w1 = w0 + crop_size[1]

    # 画像クロップ
    out_img = image.crop((h0, w0, h1, w1))

    return out_img


# 画像パッチのサイズ
patch_size = (8, 8)

# 基底の数
num_basis = 100

if SHOW_LOADED_DICT:
    # 辞書を読み込み
    V = np.load(DICT_PATH)
    out_img = np.array(V).reshape(10, 10, 8, 8)
    plt.imshow(out_img.transpose(0, 2, 1, 3).reshape(80, 80), cmap="gray")
    plt.colorbar()
    plt.title("Base Images")
    plt.show()
    exit()
else:
    # 画像パスリストの読み込み
    img_path_list = make_image_path_list(IMG_DIR)
    if len(img_path_list) == 0:
        print("No image files in {}".format(IMG_DIR))
        exit(-1)

    # 画像リスト読み込み
    if LOAD_FROM_NPY:
        imgArray = np.load(IMG_LIST_PATH)
    else:
        imgArray = ImageListFile2Array(img_path_list)
        np.save("images.npy", imgArray)

    # 辞書クラスの初期化
    print("Learning the dictionary... ")
    t0 = time()
    dico = MiniBatchDictionaryLearning(
        n_components=num_basis,
        alpha=1.0,
        transform_algorithm="lasso_lars",
        transform_alpha=1.0,
        fit_algorithm="lars",
        max_iter=1000,
        n_jobs=-1,
    )

    # 平均を0、標準偏差を1にする(白色化)
    M = np.mean(imgArray, axis=0)[np.newaxis, :]
    whiteArray = imgArray - M
    whiteArray /= np.std(whiteArray, axis=0)

    # 辞書を計算
    V = dico.fit(whiteArray).components_

    reshaped_img = np.array(V).reshape(10, 10, 8, 8)  # 画像を10x10に並べる
    plt.imshow(reshaped_img.transpose(0, 2, 1, 3).reshape(80, 80), cmap="gray")
    plt.colorbar()
    plt.title("Base Images")
    plt.savefig("images/output/Dictionaries.png")
    plt.show()

    # 辞書を保存
    np.save("Dictionaries.npy", V)
