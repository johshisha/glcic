import glob
import os
import cv2
import numpy as np
import sys

if len(sys.argv) < 2:
    mode = 'test'
else:
    mode = sys.argv[1]

ratio = 0.9
image_size = 128

x = []
paths = glob.glob('./images/*')
for path in paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x.append(img)

x = np.array(x, dtype=np.uint8)
np.random.shuffle(x)

if mode == 'train':
    p = int(ratio * len(x))
    x_train = x[:p] # Dont do this without a lot of pictures
    x_test = x[p:]
else:
    x_test = x

if not os.path.exists('./npy'):
    os.mkdir('./npy')

if mode == 'train':
    np.save('./npy/x_train.npy', x_train)
np.save('./npy/x_test.npy', x_test)
