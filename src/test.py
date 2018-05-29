import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
from network import Network
import glob
import random
IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 1  # increase this with more training data
PRETRAIN_EPOCH = 100
#test_npy = '../../data/npy/x_test.npy'


def old_to_npy():
    ratio = 0.95
    image_size = IMAGE_SIZE
    all_image_data = []
    paths = glob.glob('data/images/*')
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        all_image_data.append((path, img))
#    x = np.array(x, dtype=np.uint8)
    # np.random.shuffle(x)
    #p = int(ratio * len(x))
    # x_train = x[:p] # Dont do this without a lot of pictures
    #x_test = x
    # if not os.path.exists('./npy'):
        # os.mkdir('./npy')
    # np.save('./npy/x_train.npy', x_train)
#    np.save('./npy/x_test.npy', x_test)
    return all_image_data


def test():
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(
        tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(
        tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(
        tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])
    model = Network(x, mask, local_x, global_completion,
                    local_completion, is_training, batch_size=BATCH_SIZE)
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, 'src/backup/latest')
    #x_test = np.load(test_npy)
    datas = old_to_npy()
    random.shuffle(datas)
    paths, imgs = zip(*datas)
    x_test = np.array(imgs, dtype=np.uint8)

    np.random.shuffle(x_test)
    x_test = np.array([a / 127.5 - 1 for a in x_test])
    step_num = int(len(x_test) / BATCH_SIZE)
    cnt = 0
    for i in tqdm.tqdm(range(step_num)):
        print(paths[i])
        x_batch = x_test[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        mask_batch = get_points(paths[i])
        completion = sess.run(model.completion, feed_dict={
                              x: x_batch, mask: mask_batch, is_training: False})
        for i in range(BATCH_SIZE):
            cnt += 1
            raw = x_batch[i]
            raw = np.array((raw + 1) * 127.5, dtype=np.uint8)
            masked = raw * (1 - mask_batch[i]) + \
                np.ones_like(raw) * mask_batch[i] * 255
            img = completion[i]
            img = np.array((img + 1) * 127.5, dtype=np.uint8)
            dst = 'src/test/output/{}.png'.format("{0:06d}".format(cnt))
            output_image([['Input', masked], ['Output', img],
                          ['Ground Truth', raw]], dst)


def get_points(path):
    points = []
    point_dicts = get_point_dict(path)
    mask = []
    # mask = point_dict.values()
    for i in range(BATCH_SIZE):
        for point_dict in point_dicts.values():
            points = point_dict.values()

            for point in points:
                # point = [95, 300, 325, 31]
                print(point)
                # point = [x, y, w, h]
                BOX_SIZE = 25
                w, h = point[2], point[3]
                x1, y1 = point[0], point[1]
                x2, y2 = x1 + w, y1 + h
                # points.append([x1, y1, x2, y2])
                p1 = np.int(x1)
                q1 = np.int(y1)
                p2 = p1 + w
                q2 = q1 + h
                m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
                m[q1:q2 + 1, p1:p2 + 1] = 1
                mask.append(m)
    return np.array(mask)


def get_point_dict(image_path):
    IMAGE_FOLDER = 'images'
    COORDINATE_FOLDER = 'coordinates'
    point_path = image_path.replace(IMAGE_FOLDER, COORDINATE_FOLDER)
    point_path = point_path.replace('png', 'npy')
    point_ndarray = np.load(point_path)
    return dict(np.ndenumerate(point_ndarray))


def output_image(images, dst):
    fig = plt.figure()
    for i, image in enumerate(images):
        text, img = image
        fig.add_subplot(1, 3, i + 1)
        plt.imshow(img)
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
        plt.gca().get_xaxis().set_ticks_position('none')
        plt.gca().get_yaxis().set_ticks_position('none')
        plt.xlabel(text)
    plt.savefig(dst)
    plt.close()


if __name__ == '__main__':
    test()
