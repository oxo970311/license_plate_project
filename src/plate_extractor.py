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
# print(car_list)


# ROI 로 지정한 영역을 리스트에 저장하는 코드
save_dir = '../extracted_plates'
os.makedirs(save_dir, exist_ok=True)

for i, roi_coords in enumerate(roi_list):
    if i >= len(car_list):
        print(f"경고: 원본 이미지가 부족하여 모든 ROI를 처리할 수 없습니다. (처리된 ROI: {i}개)")
        break

    (x1, y1), (x2, y2) = roi_coords
    car_image = car_list[i]

    if y1 < 0 or y2 > car_image.shape[0] or x1 < 0 or x2 > car_image.shape[1]:
        print(f"경고: ROI {i + 1}의 좌표가 이미지 범위를 벗어납니다. ({roi_coords}) - 건너뜀")
        continue

    roi = car_image[y1:y2, x1:x2]
    save_path = os.path.join(save_dir, f'roi_{i + 1}.jpg')
    cv2.imwrite(save_path, roi)
    print(f"저장 완료: {save_path}")


def convert_to_grayscale(plate_img):
    img_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    return img_gray


def catch_edge(img):
    edges = cv2.Canny(img, 100, 400)
    # edges = cv2.Canny(img, threshold1, threshold2)  # Canny 의 threshold 값 MIN MAX 를 조절하여 선 감지를 조절 할 수 있음
    return edges


# 아래의 주소에 해당하는 이미지(ROI 이미지)를 리스트에 불러오는 코드
roi_img_path = glob.glob('../extracted_plates/*.jpg')

roi_img_list = []
for path in roi_img_path:

    temp = cv2.imread(path)
    # cv2.imshow('temp', temp)
    roi_img_list.append(temp)
    # key = cv2.waitKey(0)
    # if key == 27:  # ESC 키로 종료
    #     break

    if temp is None:
        print(f"이미지 불러오기 실패: {path}")
        continue

# ROI 로 영역을 정한 이미지를 BGR2GRAY 한 뒤 선의 갯수를 카운트 및 저장하는 코드
save_path = '../extracted_plates'
os.makedirs(save_path, exist_ok=True)
edges_list = []
for i in range(5):
    img_gray = convert_to_grayscale(roi_img_list[i])
    edges = catch_edge(img_gray)

    cv2.imshow('img_gray', img_gray)
    cv2.imshow('edges', edges)
    edge_count = cv2.countNonZero(edges)
    edges_list.append(edges)
    cv2.imwrite(f'{save_path}edge_img_{i}.jpg', edges_list[i])

    if edge_count > 20:
        print("번호판 입니다.")

    else:
        print("모르겠습니다.")
    key = cv2.waitKey(1000)  # 1000ms = 1초 동안 보여줌
    if key == 27:  # ESC 키 누르면 종료
        break

cv2.destroyAllWindows()
