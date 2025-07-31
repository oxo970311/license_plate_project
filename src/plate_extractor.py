import cv2
import numpy as np
import glob
import os
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

save_dir = '../extracted_plates'
os.makedirs(save_dir, exist_ok=True)

roi_index = 0
img_index = 0
for i, car in enumerate(car_list):
    # 예: 이미지 1장당 ROI 2개씩 있다고 가정
    rois_per_image = 2  # ← 여기서 이미지당 ROI 개수 조절

    for j in range(rois_per_image):
        if roi_index >= len(roi_list):
            break
        (x1, y1), (x2, y2) = roi_list[roi_index]
        roi = car[y1:y2, x1:x2]  # NumPy slicing

        save_path = os.path.join(save_dir, f'car_{i + 1}_roi_{j + 1}.jpg')
        cv2.imwrite(save_path, roi)
        print(f"저장 완료: {save_path}")
        roi_index += 1

# car_test = cv2.imread('../img/car_01.jpg')
# cv2.imshow('car_test', car_test)
cv2.destroyAllWindows()
