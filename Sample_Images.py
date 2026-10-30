import numpy as np
import cv2 as cv


an = 0
if an == 1:
    Original = np.load('Original.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    for i in range(3):
        ind1 = np.where(Target[i] == 0)
        ind2 = np.where(Target[i] == 1)
        for j in range(10):
            im1 = Original[i][ind1[0][i]]
            im2 = Original[i][ind2[0][i]]
            cv.imwrite('./SampleImages/Dataset-'+str(i+1)+'-Normal-'+str(j+1)+'.jpg', im1)
            cv.imwrite('./SampleImages/Dataset-' + str(i + 1) + '-Affected-' + str(j + 1) + '.jpg', im2)