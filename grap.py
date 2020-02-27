import matplotlib.pyplot as plt
from cv2 import cv2
import numpy as np
import time


def resize_image(image, ratio):
    image_shape = image.shape[:2]
    return cv2.resize(image, (int(ratio * image_shape[1]), int(ratio * image_shape[0])))


def cvt_gray(image):
    # change RGB channel to Gray
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def cvt_hsv(image):
    # change RGB channel to HSV
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def gaussian_noise(image, kernel_size):
    # Gaussian Noise Kernel
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def morph_circle(radius):
    # A circle for morphology
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))


def binary_image_process(image, method=cv2.THRESH_OTSU + cv2.THRESH_BINARY):
    # Using threshold, image become '0' and '255'
    _, bin_image = cv2.threshold(image, 0, 255, method)
    return bin_image


def morphology_process(bin_image_, radius, method=cv2.MORPH_CLOSE):
    circle_ = morph_circle(radius)
    morph_image = cv2.morphologyEx(bin_image_, method, circle_)
    return morph_image


def mask_image(image, ratio=1.0):
    image_shape = image.shape[:2]
    mask = np.zeros_like(image)
    x1 = int(ratio * image_shape[0])
    y1 = int(ratio * image_shape[1])
    x2 = image_shape[0] - x1
    y2 = image_shape[1] - y1
    mask[x1:x2, y1:y2] = 255
    return mask


def GRAY_analysis(images, TH=0.2):
    mask = mask_image(images, TH)

    plt.clf()
    plt.figure(figsize=(10, 5))
    histr_ = cv2.calcHist([images], [0], mask, [256], [0, 256])
    plt.plot(histr_[1:, :], label='gray')
    plt.title('HISTOGRAM-GRAY-{}.png'.format(num + 1))
    plt.xlim([0, 256])
    plt.legend()
    plt.grid()
    plt.savefig('./HISTOGRAM-GRAY-{}.png'.format(num + 1))


def draw_box(binary_image, init_image):
    # 產生等高線
    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 建立除錯用影像
    img_debug = init_image.copy()

    # 線條寬度
    line_width = int(init_image.shape[1] / 255)

    # 以藍色線條畫出所有的等高線
    # cv2.drawContours(img_debug, contours, -1, (255, 0, 0), line_width)

    # 找出面積最大的等高線區域
    c = max(contours, key=cv2.contourArea)

    # 找出可以包住面積最大等高線區域的方框，並以綠色線條畫出來
    x, y, w, h = cv2.boundingRect(c)

    outs = cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), line_width)
    cv2.imwrite('../images/Segmet_Images/box_Green.png', outs)

    # 嘗試在各種角度，以最小的方框包住面積最大的等高線區域，以紅色線條標示
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    Fin_image = cv2.drawContours(img_debug, [box], 0, (0, 0, 255), line_width)
    cv2.imwrite('../images/Segmet_Images/box_Rad.png', Fin_image)


if __name__ == '__main__':
    tic = time.time()
    circle = morph_circle(3)
    num = 6

    path = '../images/7A17-190_/{}.png'.format(num)

    src_image = cv2.imread(path)
    src_image = resize_image(src_image, 0.2)
    hsv_image = cvt_hsv(src_image)
    gray_image = cvt_gray(src_image)

    img_green_channel = src_image[:, :, 1]
    img_saturation = hsv_image[:, :, 1]
    img_value = hsv_image[:, :, 2]

    green = cv2.equalizeHist(img_green_channel)
    gray = cv2.equalizeHist(gray_image)

    img_gray_GN = gaussian_noise(gray, 5)
    img_sat_GN = cv2.GaussianBlur(img_saturation, (5, 5), 0)
    img_v_GN = cv2.GaussianBlur(img_value, (5, 5), 0)
    img_g_GN = cv2.GaussianBlur(img_green_channel, (5, 5), 0)

    alpha = 0.7
    beta = 1 - alpha

    out = cv2.addWeighted(img_sat_GN, alpha, gray, beta, 0)
    cv2.imshow("Add image", out)
    out = cv2.equalizeHist(out)
    cv2.imshow("Add image with eq", out)

    bin_ = binary_image_process(out, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    bin_ = morphology_process(bin_, 5, cv2.MORPH_OPEN)
    bin_ = morphology_process(bin_, 7, cv2.MORPH_CLOSE)

    # bin_ = cv2.erode(bin_, circle, iterations=2)
    # bin_ = cv2.dilate(bin_, circle, iterations=2)
    # bin_ = cv2.erode(bin_, circle, iterations=2)

    cv2.imshow("Image", bin_)
    cv2.waitKey()
    print('# Time : %.3f' % (time.time() - tic))
