import cv2 as cv
import numpy as np

# 读取图片、添加标签并保存成csv
n_class = 15
n_samples = 11
samples = []
for class_ID in range(n_class):
    data_tmp = []
    for img_id in range(n_samples):
        img = cv.imread('./Yale/' + str(class_ID + 1) + '/s' + str(img_id + 1) + '.bmp', cv.IMREAD_GRAYSCALE)
        img = np.reshape(img, (1, -1))[0]
        data_tmp.append(img)
    data_tmp = np.array(data_tmp)
    labels = (class_ID + 1) * np.ones((n_samples, 1), dtype=np.int32)
    data_tmp = np.concatenate((data_tmp, labels), axis=1)
    samples.append(data_tmp)
X = samples[0]
for i in range(n_class - 1):
    X = np.concatenate((X, samples[i + 1]), axis=0)
np.savetxt('Yale.csv', X, delimiter=',', fmt='%d')
