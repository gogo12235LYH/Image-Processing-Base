import cv2
import numpy as np


def main():
    NUM = 5
    src = cv2.imread("D:\\Mark_Area{}.png".format(NUM))
    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1.5)

    h, w = img.shape[:2]
    ratio = 1.0
    img = cv2.resize(img, (int(ratio * w), int(ratio * h)))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    img = cv2.Canny(img, 50, 150, apertureSize=5)

    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, 20, 25)
    cv2.imshow("Image", img)
    cv2.waitKey()
    print(lines[:2], img.shape)

    min_x = int(min(lines[:, :, 0])) + 1
    min_y = int(min(lines[:, :, 1])) + 1
    max_x = int(max(lines[:, :, 0])) + 1
    max_y = int(max(lines[:, :, 1])) + 1

    # cx = int((max_x + min_x) / 2) + 1
    # cy = int((max_y + min_y) / 2) + 1
    # roi = cv2.line(src, (cx, 0), (cx, cy + 100), (0, 255, 0), 2)
    # roi = cv2.line(roi, (0, cy), (cx + 100, cy), (0, 255, 0), 2)
    # roi = cv2.circle(roi, (cx, cy), 5, (0, 0, 255), -1)
    # roi = roi[min_y:max_y, min_x:max_x, :]

    roi = src[min_y:max_y, min_x:max_x, :]
    cv2.imwrite("D:\\MarkPoint\\MarkPointTemplate{}.png".format(NUM), src[min_y:max_y, min_x:max_x, :])
    cv2.imshow("ROI", roi)
    cv2.waitKey()


if __name__ == "__main__":
    main()
