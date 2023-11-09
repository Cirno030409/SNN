import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

coefs = np.load("Coefficients.npy")
dic = np.load("Dictionaries.npy")

patch = 467



for i in coefs[patch]:
    coefs_list = i.tolist()
    
for i in range(len(coefs_list)):
    if coefs_list[i] != 0:
        print("base_num", i+1, "coefs_num", coefs_list[i])
        # print(dic[i])
        

