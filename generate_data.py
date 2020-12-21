import numpy as np
from glob import glob
from tqdm import tqdm
import PIL  

def generate_arrays_from_file(path,batch_size):
    while 1:
        img_size = 400  
        cnt = 0
        X =[]
        Y =[]
        for img_path in tqdm(glob(path + '/*/*.jpg')):
            img = PIL.Image.open(img_path)
            img = img.resize((img_size, img_size))  # 图片resize
            arr = np.asarray(img)  # 图片转array
            X.append(arr)  # 赋值
            if img_path.split('\\')[-2] == 'SSAP':
                Y.append([0,1])
            else:
                Y.append([1,0])
            cnt += 1
            if cnt==batch_size:
                cnt = 0
                yield (np.array(X), np.array(Y))
                X = []
                Y = []