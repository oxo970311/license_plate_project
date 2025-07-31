import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

img_path = glob.glob('../img/car_*.jpg')

drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
roi_list = []
car_list = []

def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        roi_list.append((start_point, end_point))
        print(f"선택된 ROI: {start_point} → {end_point}")


for path in img_path:
    temp = cv2.imread(path)
    if temp is None:
        print(f"이미지 불러오기 실패: {path}")
        continue

    clone = temp.copy()
    cv2.namedWindow('temp')
    cv2.setMouseCallback('temp', draw_rectangle)

    while True:
        display_img = clone.copy()
        if drawing:
            cv2.rectangle(display_img, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow('temp', display_img)
        key = cv2.waitKey(1)

        if key == 13:  # Enter 키 누르면 다음 이미지로
            car_list.append(temp)
            break
        elif key == 27:  # ESC 누르면 종료
            exit()

print(roi_list)

# car_test = cv2.imread('../img/car_01.jpg')
# cv2.imshow('car_test', car_test)
cv2.destroyAllWindows()
