import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os.path as osp

all_res = glob.glob('results/*.png')
all_res = sorted(all_res)

for path in all_res:
    plt.figure()
    img = np.array(Image.open(path))
    plt.imshow(img)
    plt.title(osp.basename(path))
    plt.show()