import numpy as np
import cv2
import cv2 as  cv
from FCM import FCM
from Region_Growing import Region_Growing
import numpy as np
import cv2 as cv
from collections import deque


def regionGrow(img, seeds, thresh):
    rows, cols = img.shape
    visited = np.zeros((rows, cols), dtype=bool)
    region = np.zeros((rows, cols), dtype=np.uint8)
    queue = deque()

    for x, y in seeds:
        if 0 <= x < cols and 0 <= y < rows:
            if 50 <= int(img[y, x]) <= 240:
                queue.append((x, y))
                visited[y, x] = True
                region[y, x] = 255

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        current_value = int(img[y, x])

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < cols and 0 <= ny < rows:
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    neighbor_value = int(img[ny, nx])

                    if abs(neighbor_value - current_value) <= thresh:
                        if 50 <= neighbor_value <= 240:
                            region[ny, nx] = 255
                            queue.append((nx, ny))

    return region


def model_region_growing(original, sol=None):
    if sol is None:
        sol = [50, 5]
    Seg_Img = []
    seed_points = [(10, 10), (82, 120), (20, 100)]
    threshold_limit = int(sol[1])  # Choose a value from 5 to 25
    for j in range(len(original)):
        print(j, len(original))
        gray = cv.cvtColor(
            original[j], cv.COLOR_BGR2GRAY
        )

        # Step 1: Region Growing
        grown_region = regionGrow(gray, seed_points, threshold_limit)

        # Step 2: Threshold the grown region
        masked_image = cv.bitwise_and(
            gray, gray, mask=grown_region
        )

        _, binaryImg = cv.threshold(
            masked_image, 127, 255, cv.THRESH_BINARY
        )
        Seg_Img.append(binaryImg)
    return Seg_Img


def Model_Regiongrowing(Feat, sol=None):
    if sol is None:
        sol = [50, 5]

    seed = int(sol[0])
    Threshhold = int(sol[1])
    Images = Feat
    Seg_Img = []
    for j in range(len(Images)):  # len(Images)
        print(j, len(Images))
        original_image = Images[j]
        Gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        image = Gray_img.copy()
        cluster = FCM(image, image_bit=4, n_clusters=5, m=10, epsilon=0.8, max_iter=35)
        cluster.form_clusters()
        result = cluster.result.astype('uint8') * 50
        # cv2.imshow('Images', image)
        # cv2.imshow('FCM Image', result)  # Seg_Image
        # cv2.waitKey(0)
        index_img = np.where(image == np.max(image))
        label = result[index_img[1]]
        uniq, counts = np.unique(label, return_counts=True)
        cc_sort = np.argsort(counts)
        out_img = np.zeros(image.shape)
        index_1 = np.where(result == uniq[cc_sort[len(cc_sort) - 1]])
        out_img[index_1[0], index_1[1]] = 255
        out_img = out_img.astype('uint8')
        seed = [index_1[0][index_1[0].shape[0] // 2], index_1[1][index_1[1].shape[0] // 2]]
        # seed = (Excel[j, 5], Excel[j, 4])
        reg_grow = Region_Growing(image, seed)
        fuzzy_img = 0.5 * out_img
        reg_img = 0.5 * reg_grow
        reg_img = reg_img[:fuzzy_img.shape[0], :fuzzy_img.shape[1]]
        fuzzy_ind = np.where(fuzzy_img != 0)
        fuzzy_ind = fuzzy_ind[0] * image.shape[1] + fuzzy_ind[1]
        reg_ind = np.where(reg_img != 0)
        reg_ind = reg_ind[0] * image.shape[1] + reg_ind[1]

        index = np.intersect1d(fuzzy_ind, reg_ind)
        result_image = np.zeros(image.shape)
        result_image[index // result_image.shape[1], index % result_image.shape[1]] = 255
        Seg_Img.append(result_image)

    return np.asarray(Seg_Img)

