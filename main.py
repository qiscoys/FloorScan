import cv2
import numpy as np
import pytesseract
import re
import time
import os
import json
from matplotlib import pyplot as plt
import difflib

# ------------------房间颜色定义------------
ROOM_COLORS = {#(B,G,R)
    "KITCHEN": (203, 192, 255),        # 粉
    "YARD": (192, 192, 192),         # 银色
    "MASTER BEDROOM": (181, 228, 255),  # 莫卡辛
    "DINING": (127, 255, 212),         # 绿色
    "BEDROOM": (181, 228, 253),      # 莫卡辛
    "BATH": (222, 196, 176),         # 浅钢蓝
    "LIVING": (216, 191, 216),       # 蓟色
    "BALCONY": (152, 251, 152),      # 淡绿色
    "MASTER BATH": (222, 196, 176),# 浅钢蓝
    "FOYER": (245, 240, 255),      # 薰衣草红色
    "STUDY ROOM": (255, 215, 0),   # 金色
    "WET KITCHEN": (71, 99, 255),          # 番茄色
    "YARD WET KITCHEN": (71, 99, 255),  # 番茄色
    "UTILITY": (71, 99, 255),         # 番茄色

}

# ------------------提取房间标签------------
def extract_room_labels(img):

    # 创建一个与输入图像相同大小的掩膜，初始值为0（黑色）
    mask = np.zeros(img.shape, dtype=np.uint8)

    # 将输入图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用大津法进行二值化
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 查找轮廓
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历找到的轮廓
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 120 and 8 < h < 40: 
            # 在掩膜上绘制蓝色轮廓
            cv2.drawContours(mask, [cnt], 0, (255, 0, 0), 1)

    # 创建一个4x4的结构元素，用于膨胀操作
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)

    # 将膨胀后的掩膜转换为灰度图
    gray_d = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)

    # 对膨胀后的图像进行二值化
    threshold_d = cv2.threshold(gray_d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 查找膨胀后图像的轮廓
    contours_d, _ = cv2.findContours(threshold_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建两个副本，用于后续处理和展示
    text_extracted_img = img.copy()
    bounded_text_img = img.copy()
    ROI = []          
    coordinates = []  

    # 提取房间标签的区域
    for cnt in contours_d:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 10:
            cv2.rectangle(bounded_text_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_c = bounded_text_img[y:y + h, x:x + w]
            coordinates.append((x, y, w, h))
            ROI.append(roi_c)
            pad = 10
            y1 = max(0, y - pad)
            y2 = min(img.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(img.shape[1], x + w + pad)

            text_extracted_img[y1:y2, x1:x2] = 255

    room_labels = []  

    #把ROI输入Tesseract，识别文字标签
    for i, room in enumerate(ROI):
        text = pytesseract.image_to_string(room, lang="chi_sim+eng", config="--psm 6")
        text = text.replace("\n", ' ').strip()
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9 ]', '', text)
        if text:
            room_labels.append((text, coordinates[i]))
            
    return room_labels, text_extracted_img

# ------------------填补墙壁间隙------------
def fill_wall_gaps(img):
    corners_detacted_img = img.copy()  
    gaps_filled_img = img.copy()  

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = np.float32(thresh)

    # 使用 Harris 角点检测算法查找角点
    dst = cv2.cornerHarris(thresh, 3, 7, 0.07)
    dst = cv2.dilate(dst, None)
    corners_detacted_img[dst > 0.1 * dst.max()] = [255, 0, 0]  

    corners = dst > 0.1 * dst.max()
    
    # 填补墙壁间隙，分割线改为黑色
    for y, row in enumerate(corners):
        x_same_y = np.argwhere(row)
        for x1, x2 in zip(x_same_y[:-1], x_same_y[1:]):
            if abs(x2[0] - x1[0]) < 70:
                cv2.line(gaps_filled_img, (int(x1[0]), int(y)), (int(x2[0]), int(y)), (0, 0, 0), 2)  

    for x, col in enumerate(corners.T):
        y_same_x = np.argwhere(col)
        for y1, y2 in zip(y_same_x[:-1], y_same_x[1:]):
            if abs(y2[0] - y1[0]) < 70:
                cv2.line(gaps_filled_img, (int(x), int(y1[0])), (int(x), int(y2[0])), (0, 0, 0), 2)  
                
    return gaps_filled_img

# ------------------检测房间区域------------
def detect_rooms(orignial_img,gaps_filled_img, text_extracted_img):
    segmented_rooms_img = text_extracted_img.copy()
    gray = cv2.cvtColor(gaps_filled_img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((3, 3), np.uint8)
    mor_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2) 

    contours, _ = cv2.findContours(mor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_img2 = orignial_img.copy()
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=False)

    room_coordinates = []
    initial_contours = []
    img_size = gaps_filled_img.shape[0] * gaps_filled_img.shape[1]

    for c in sorted_contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if area > 1000 and area < img_size * 0.5 and w * h < img_size: 
            initial_contours.append(c)
            cv2.drawContours(contours_img2, [c], -1, (0, 255, 0), 2)  
            
    return room_coordinates, initial_contours, segmented_rooms_img

# ------------------匹配房间与标签并绘制房间名------------
def match_room_and_label(room_labels, initial_contours, img):
    final_segmented_rooms_img = img.copy()
    final_list = []

    font_scale = 0.5  
    font_thickness = 2  
    vertical_shift = 20  

    CHINESE_TO_ENGLISH = {
        "厨房": "KITCHEN",
        "主卧": "MASTER BEDROOM",
        "卧室": "BEDROOM",
        "次卧": "BEDROOM",
        "餐厅": "DINING",
        "卫生间": "BATH",
        "卫": "BATH",
        "洗手间": "BATH",
        "主卫": "MASTER BATH",
        "客卫": "BATH",
        "客厅": "LIVING",
        "厅": "LIVING",
        "阳台": "BALCONY",
        "走廊": "FOYER",
        "书房": "STUDY ROOM",
        "院子": "YARD"
    }

    for label in room_labels:
        label_midpoint = [((2 * label[1][0] + label[1][2]) / 2),
                          ((2 * label[1][1] + label[1][3]) / 2)]
        room_name = label[0].split(':')[0].strip().upper()  
        
        best_contour = None
        max_dist = -float('inf')
        
        for c in initial_contours:
            dist = cv2.pointPolygonTest(c, tuple(map(int, label_midpoint)), True)
            if dist > max_dist:
                max_dist = dist
                best_contour = c
                
        if best_contour is not None and max_dist >= -20:
            c = best_contour
            x, y, w, h = cv2.boundingRect(c)
            
            matched_room_name = None
            
            for ch_key, en_val in CHINESE_TO_ENGLISH.items():
                if ch_key in room_name:
                    matched_room_name = en_val
                    break
            
            if not matched_room_name:
                valid_room_names = list(ROOM_COLORS.keys())
                matches = difflib.get_close_matches(room_name, valid_room_names, n=1, cutoff=0.6)
                if matches:
                    matched_room_name = matches[0]
            
            if matched_room_name: 
                color = ROOM_COLORS.get(matched_room_name, (200, 200, 200))
                
                epsilon = 0.005 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                
                cv2.fillPoly(final_segmented_rooms_img, [approx], color)
                polygon = approx.reshape(-1, 2).tolist()
                final_list.append({
                    "name": matched_room_name, 
                    "box": (x, y, w, h),
                    "polygon": polygon
                })

                text_position = (label[1][0], label[1][1] + vertical_shift)

                if len(matched_room_name) > 10:  
                    first_line = matched_room_name[:10]  
                    second_line = matched_room_name[10:]  
                    cv2.putText(final_segmented_rooms_img, first_line, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                    second_line_position = (text_position[0], text_position[1] + 15)
                    cv2.putText(final_segmented_rooms_img, second_line, second_line_position,
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                else:
                    cv2.putText(final_segmented_rooms_img, matched_room_name, text_position,
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

    return final_list, final_segmented_rooms_img

def extract_walls_data(img):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([60, 60, 60])
    black_mask = cv2.inRange(img, lower_black, upper_black)

    kernel_open = np.ones((3, 3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(black_mask)
    wall_thresh = np.zeros_like(black_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 400:
            wall_thresh[labels == i] = 255

    kernel_close = np.ones((5, 5), np.uint8)
    wall_thresh = cv2.morphologyEx(wall_thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, hierarchy = cv2.findContours(wall_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    walls_data = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, c in enumerate(contours):
            if hierarchy[i][3] == -1:
                area = cv2.contourArea(c)
                if area > 300:
                    epsilon = 0.001 * cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, epsilon, True)
                    outer_polygon = approx.reshape(-1, 2).tolist()

                    holes = []
                    child_idx = hierarchy[i][2]
                    while child_idx != -1:
                        child_c = contours[child_idx]
                        if cv2.contourArea(child_c) > 50:
                            c_epsilon = 0.001 * cv2.arcLength(child_c, True)
                            c_approx = cv2.approxPolyDP(child_c, c_epsilon, True)
                            holes.append(c_approx.reshape(-1, 2).tolist())
                        child_idx = hierarchy[child_idx][0]

                    walls_data.append({
                        "outer": outer_polygon,
                        "holes": holes
                    })
    return walls_data

# ------------------主程序------------
def create_video(output_path, image_files, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  
    first_frame = cv2.imread(image_files[0])
    height, width, layers = first_frame.shape
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        if img is not None:
            video_writer.write(img)
    video_writer.release()

if __name__ == "__main__":
    input_folder = './test_data/'
    output_folder = './final_data/'
    compare_folder = './testVSfinal/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)

    image_files = sorted(
        [
            os.path.join(input_folder, file_name)
            for file_name in os.listdir(input_folder)
            if file_name.lower().endswith('.png') and os.path.isfile(os.path.join(input_folder, file_name))
        ]
    )
    result_image_files = []

    start_time = time.time()  

    for image_file in image_files:

        img = cv2.imread(image_file)
        if img is None:
            print(f"Error loading image: {image_file}")
            continue

        room_labels, text_extracted_img = extract_room_labels(img)
        print("Room labels: " + str(room_labels))

        gaps_filled_img = fill_wall_gaps(text_extracted_img)

        room_coordinates, initial_contours, segmented_rooms_img = detect_rooms(img, gaps_filled_img, text_extracted_img)

        matched_rooms, final_segmented_rooms_img = match_room_and_label(room_labels, initial_contours, segmented_rooms_img)

        # 导出第一张图的房间多边形数据供 Three.js 使用
        if len(result_image_files) == 0 and matched_rooms:
            with open("rooms_polygon.json", "w", encoding="utf-8") as f:
                json.dump(matched_rooms, f, ensure_ascii=False, indent=4)
            walls_data = extract_walls_data(img)
            with open("walls_polygon.json", "w", encoding="utf-8") as f:
                json.dump(walls_data, f, ensure_ascii=False, indent=4)

        file_name = os.path.basename(image_file)
        output_path = os.path.join(output_folder, file_name)
        compare_path = os.path.join(compare_folder, file_name)
        cv2.imwrite(output_path, final_segmented_rooms_img)
        compare_img = np.hstack((img, final_segmented_rooms_img))
        cv2.imwrite(compare_path, compare_img)
        result_image_files.append(output_path)

    end_time = time.time()  

    total_time = end_time - start_time  
    processed_count = len(result_image_files)
    average_time = total_time / processed_count if processed_count else 0

    print(f"Total time for processing {processed_count} images: {total_time:.2f} seconds")
    print(f"Average time per image: {average_time:.2f} seconds")
    if processed_count:
        create_video('./result.avi', result_image_files, fps=2)
'''
if __name__ == "__main__":
    plt.figure()
    img = cv2.imread('./test_data/32.png')  # 读取图像
    cv2.imshow('Original Image', img)  # 显示原始图像
    if img is None:
        print("Error loading image")
    else:
        # 提取房间标签
        room_labels, text_extracted_img = extract_room_labels(img)
        print("Room labels: " + str(room_labels))

        # 填补墙壁间隙
        gaps_filled_img = fill_wall_gaps(text_extracted_img)

        # 检测房间区域
        room_coordinates, initial_contours, segmented_rooms_img = detect_rooms(img,gaps_filled_img, text_extracted_img)
        print("\nRooms coordinates: " + str(room_coordinates))

        # 匹配房间与标签
        final_list, final_segmented_rooms_img = match_room_and_label(room_labels, initial_contours, text_extracted_img)
        print("\nFinal output: " + str(final_list))

    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 关闭所有窗口
'''
