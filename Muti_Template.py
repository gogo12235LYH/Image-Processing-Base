import cv2
import numpy as np
import time


def multi_template(image, temp):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    res_ = cv2.matchTemplate(img_gray, temp_gray, cv2.TM_CCOEFF_NORMED)

    w, h = temp_gray.shape[:2]

    threshold = 0.75
    loc = np.where(res_ >= threshold)
    print("# Check ~")

    pixel_ = 50

    T_count = 4

    th1, th2, cnt = 0, 0, 0
    pt_data = []
    F_i = len(loc[0])-2

    for i in range(len(loc[0])-1):
        check1 = abs(loc[0][i] - loc[0][i+1])
        check2 = abs(loc[1][i] - loc[1][i+1])

        if check1 > pixel_ and check2 > pixel_:
            th1 = i + 1
            cnt += 1
            # print("# ", check1, " < = > ", th2, " - ", th1, "  | Current Counts : ", cnt)
            new_pt = tuple([int(np.mean(loc[1][th2:th1])), int(np.mean(loc[0][th2:th1]))])
            pt_data.append(new_pt)
            th2 = th1

        if cnt + 1 == T_count or i == F_i:
            new_pt = tuple([int(np.mean(loc[1][th2:])), int(np.mean(loc[0][th2:]))])
            pt_data.append(new_pt)
            # print("# ", check1, " < = > ", th2, " - End", " | Current Counts : ", cnt+1)
            print("# Done ~")
            break

    for i in range(len(pt_data)):
        cv2.rectangle(image, pt_data[i], (pt_data[i][0] + w, pt_data[i][1] + h), (0, 0, 255), 2)
        print("# Mark Point Info : (", pt_data[i], ")")

    return image


def resize_img(image_, ratio=1.0):
    h, w = image_.shape[:2]
    image_ = cv2.resize(image_, (int(ratio * w), int(ratio * h)))
    return image_


if __name__ == "__main__":
    # obj_image1 = cv2.imread("D:\\temp5.png")
    obj_image2 = cv2.imread("D:\\FF_.png")
    temp_image = cv2.imread("D:\\MarkPoint\\MarkPointTemplate1.png")

    tic = time.time()

    # obj_image1 = resize_img(obj_image1, 0.4)
    obj_image2 = resize_img(obj_image2, 0.4)
    temp_image = resize_img(temp_image, 0.4)

    # out_image1 = multi_template(obj_image1, temp_image)
    out_image2 = multi_template(obj_image2, temp_image)
    print("# Total Spent Times : %.3f ms" % ((time.time() - tic)*1000))

    # out_image1 = resize_img(out_image1)
    out_image2 = resize_img(out_image2)
    # cv2.imshow("Out1", out_image1)
    cv2.imshow("Out2", out_image2)
    # cv2.imwrite("D:\\out1.png", out_image1)
    cv2.imwrite("D:\\out2.png", out_image2)
    cv2.waitKey()
