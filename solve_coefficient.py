import pprint
from time import time

import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import SparseCoder
from tqdm import tqdm

DICT_PATH = "Dictionaries.npy"  # 読み込む辞書のパス
SRC_IMG_PATH = "../images/input/input.jpg"  # 構成元となる入力画像のパス

ORG_IMG_PATH = "../images/output/original.jpg"  # 構成元となる画像を保存するパス
DIST_IMG_PATH = "../images/output/reconstruction.jpg"  # 再構成した画像のパス

num_dict = 100

coefs = []


# グレースケールの配列をRGBに変換
def GrayArray2RGB(gray_array):
    gray_shape = gray_array.shape
    rgb_data = np.zeros((gray_shape[0], gray_shape[1], 3), dtype=gray_array.dtype)
    for c in range(3):
        rgb_data[:, :, c] = gray_array[:, :, 0]
    return rgb_data


# 作成した辞書をロード
V = np.load(DICT_PATH)

# 画像パッチのサイズ
patch_size = (8, 8)

# 画像を読み込み
im = Image.open(SRC_IMG_PATH)

im.save(ORG_IMG_PATH)

# グレースケール変換
gray_im = ImageOps.grayscale(im)

# 出力画像を初期化
dst_array = np.zeros((gray_im.size[1], gray_im.size[0]))

# 画像をpatch_sizeで分割して処理
w = gray_im.size[0] - patch_size[0]
h = gray_im.size[1] - patch_size[1]

save_img_num = 50


print("dst image size h: {0}, w: {1}".format(gray_im.size[1], gray_im.size[0]))
print("reconstructing...")
patch_cnt = 0
with tqdm(total=w / patch_size[0] * h / patch_size[1]) as pbar:
    y = 0
    while y <= h:
        x = 0
        while x <= w:
            pbar.update(1)
            # print("x: {0}, y: {1}".format(x, y))
            # パッチサイズの領域を切り取り
            box = (x, y, x + patch_size[0], y + patch_size[1])
            crop_im = gray_im.crop(box)

            # arrayに格納
            data = np.asarray(crop_im)
            data = data.reshape(1, data.size)

            # Sparse Coding
            coder = SparseCoder(
                dictionary=V,
                transform_algorithm="omp",
                transform_alpha=0.1,
                transform_n_nonzero_coefs=10,
                transform_max_iter=1000,
                n_jobs=-1,
            )
            u = coder.transform(data)  # 係数を計算
            coefs.append(u)  # 係数をリストに格納

            # 信号を復元
            s = np.dot(u, V)

            # if patch_cnt == save_img_num:
            crop_im.save("images/crops_raw/{0}({1},{2}).jpg".format(patch_cnt, x, y))

            crop_im2 = s.reshape(patch_size[1], patch_size[0], 1)
            crop_im2_rgb = GrayArray2RGB(crop_im2)
            crop_im2_rgb = Image.fromarray(np.uint8(crop_im2_rgb))
            crop_im2_rgb.save(
                "images/crops_processed/{0}({1},{2}).jpg".format(patch_cnt, x, y)
            )
            # np.save("coef" + str(save_img_num) + ".npy", u)

            # 復元した画像をコピー
            s = s.reshape(patch_size[1], patch_size[0])  # 画像サイズにreshape
            dst_array[y : y + patch_size[1], x : x + patch_size[0]] = s  # 画像をコピー
            x += patch_size[0]
            patch_cnt += 1
        y += patch_size[1]

# 再構成した画像を保存
print("saving...")
dst_array = dst_array.reshape(gray_im.size[1], gray_im.size[0], 1)
rgb_data = GrayArray2RGB(dst_array)
im = Image.fromarray(np.uint8(rgb_data))
im.save(DIST_IMG_PATH)
np.save("Coefficients.npy", np.array(coefs))
im.show()
