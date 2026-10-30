import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image



def Image_Results():
    Original = np.load('Original.npy', allow_pickle=True)
    Seg = np.load('Seg.npy', allow_pickle=True)
    Img_Ind = [2, 11, 15] # 6, 7, 15
    for i in range(len(Original)):
        for j in range(len(Img_Ind)):
            Se = cv.cvtColor(Seg[i][j], cv.COLOR_BGR2GRAY )
            Out = np.zeros((Se.shape)).astype('uint8')
            ind = np.where(Se > 0)
            Out[ind[0], ind[1]] = 255
            fig, ax = plt.subplots(1, 2)
            plt.suptitle('Image %d' % (j + 1), fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Orig')
            plt.imshow(Original[i][Img_Ind[j]])
            plt.subplot(1, 2, 2)
            plt.title('Segmented')
            plt.imshow(Out)
            fig.tight_layout()
            path = "./Results/Image_Res/Image_%s_%s.png" % (str(i + 1), str(j+1))
            plt.savefig(path)
            plt.show()
            cv.imwrite('./Results/Image_Res/Orig-'+str(i+1)+'-'+str(j+1)+'.jpg', Original[i][Img_Ind[j]])
            cv.imwrite('./Results/Image_Res/Seg-' + str(i + 1) + '-' + str(j + 1) + '.jpg', Out)

Image_Results()