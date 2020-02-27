import cv2
import numpy as np


def cvt_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_noise(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 2.0)


def binary(img, method=cv2.THRESH_BINARY + cv2.THRESH_OTSU):
    _, img = cv2.threshold(img, 0, 255, method)
    return img


def morphology(img, radius, method=cv2.MORPH_CLOSE):
    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    return cv2.morphologyEx(img, method, circle)


def Hough_LineP(canny_img, rho=1., theta=np.pi/180, threshold=25, minLineLength=25, maxLineGap=15):
    lines = cv2.HoughLinesP(canny_img, rho, theta, threshold, minLineLength, maxLineGap)
    return np.reshape(lines, (-1, 4))


def drawing_line(img, lines):
    for line in lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), [0, 0, 255], 2)
    return img


def process_image(src_image):
    src_image_copy = src_image.copy()
    gray = cvt_gray(src_image)
    gray_GN = gaussian_noise(gray, 3)
    # bin_image = binary(gray_GN)
    # bin_image_morph = morphology(bin_image, 7)
    edge_image = cv2.Canny(gray_GN, 150, 210)
    edge_image = morphology(edge_image, 3)
    # edge_image = morphology(edge_image, 3, cv2.MORPH_OPEN)

    cv2.imshow("Edge Image", edge_image)
    cv2.waitKey()

    lines = Hough_LineP(edge_image)

    src_image_line = drawing_line(src_image, lines)
    cv2.imshow("Src Image with lines", src_image_line)
    cv2.waitKey()

    print(lines.shape, src_image_line.shape)

    min_x = int(min(lines[:, 0])) + 1
    min_y = int(min(lines[:, 1])) + 1
    max_x = int(max(lines[:, 0])) + 1
    max_y = int(max(lines[:, 1])) + 1

    Fin_image = src_image_copy[min_y:max_y, min_x:max_x, :]
    cv2.imwrite("D:\\AOI_STI\\images\\MarkPoint\\MarkPointTemplate{}.png".format(NUM), Fin_image)
    cv2.imshow("ROI", Fin_image)
    cv2.waitKey()


if __name__ == "__main__":
    NUM = 25
    src = cv2.imread("D:\\AOI_STI\\images\\MarkPoint\\o_{}.png".format(NUM))
    process_image(src)
