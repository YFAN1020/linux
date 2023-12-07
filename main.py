import json
import mimetypes
import os

import cv2
import numpy as np
import requests
from paddleocr import PaddleOCR

host = 'http://192.168.0.17:8080/srai/upload/file'
image_path = "D:/Paddle_ocr_project/picture/condensor2.jpg"
images = cv2.imread(image_path)
w = 800
h = 600
image = cv2.resize(images, (w, h))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
medianblur_image = cv2.medianBlur(gray_image, 7)  # 7
_, binary_image = cv2.threshold(medianblur_image, 80, 255, cv2.THRESH_BINARY)  # 80,255

canny_image = cv2.Canny(binary_image, 100, 200)  # 100,200

contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

line_image = np.zeros_like(canny_image, dtype=np.uint8)

for index in range(len(contours)):
    color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
    cv2.drawContours(line_image, contours, index, color, 1, 8)

cv2.imshow('Line Image', line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 寻找面积最大的轮廓
max_area_index = 0
max_area = 0

for index, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_area_index = index

# 多边形逼近
epsilon = 10
closed = True
poly_approx = cv2.approxPolyDP(contours[max_area_index], epsilon, closed)
# print("poly_approx", poly_approx)
# 绘制
polyPic = np.zeros_like(image, dtype=np.uint8)
color = (0, 0, 255)
thickness = 2
cv2.drawContours(polyPic, [poly_approx], -1, color, thickness)
cv2.imshow("Poly Approx Image", polyPic)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 寻找凸包
hull = cv2.convexHull(poly_approx)
# 将凸包点绘制在图像上
for i in range(len(hull)):
    cv2.circle(polyPic, tuple(hull[i][0]), 10,
               (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)), 3)

cv2.addWeighted(polyPic, 0.5, image, 0.5, 0, image)
# 显示结果图像
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
new_poly_approx = []
for i in poly_approx:
    for x in i:
        new_poly_approx.append(x)
# print(new_poly_approx)
# 使用列表推导式将NumPy数组转换为元组
lists = [arr.tolist() for arr in new_poly_approx]
# print(lists)
# 定义源和目标点的坐标
srcPoints = [[0, 0], [0, 0], [0, 0], [0, 0]]
dstPoints = [(0, 0), (image.shape[1] - 1, 0), (image.shape[1] - 1, image.shape[0] - 1),
             (0, image.shape[0] - 1)]
# print("dst:", dstPoints)
# print("image.shape:", image.shape)
# 对四个点进行排序，分为左上、右上、右下、左下
sorted = False
n = 4
while not sorted:
    for i in range(1, n):
        sorted = True
        if lists[i - 1][0] > lists[i][0]:
            lists[i - 1], lists[i] = lists[i], lists[i - 1]
            sorted = False
    n -= 1

if lists[0][1] < lists[1][1]:
    srcPoints[0] = lists[0]
    srcPoints[3] = lists[1]
else:
    srcPoints[0] = lists[1]
    srcPoints[3] = lists[0]

if lists[2][1] < lists[3][1]:
    srcPoints[1] = lists[2]
    srcPoints[2] = lists[3]
else:
    srcPoints[1] = lists[3]
    srcPoints[2] = lists[2]

srcPoints = np.float32(srcPoints)
dstPoints = np.float32(dstPoints)

srcPoints = np.array(srcPoints, dtype=np.float32)
dstPoints = np.array(dstPoints, dtype=np.float32)
# 获取变换矩阵
transMat = cv2.getPerspectiveTransform(srcPoints, dstPoints)

# print("transMat", transMat)
# 进行坐标变换
outPic = cv2.warpPerspective(image, transMat, (image.shape[1], image.shape[0]))
cv2.imshow("outPic", outPic)
cv2.waitKey(0)

image = outPic

# 定义矩形区域的坐标
# x1, y1, w1, h1 = 10, 40, 530, 40
x2, y2, w2, h2 = 650, 15, 100, 60
x3, y3, w3, h3 = 10, 75, 250, 380
x4, y4, w4, h4 = 250, 75, 250, 380
x5, y5, w5, h5 = 500, 75, 270, 380
# 创建一个与图像大小相同的空白图像
# mask1 = np.zeros_like(image)
mask2 = np.zeros_like(image)
mask3 = np.zeros_like(image)
mask4 = np.zeros_like(image)
mask5 = np.zeros_like(image)
# 在掩膜上绘制矩形
# cv2.rectangle(mask1, (x1, y1), (x1 + w1, y1 + h1), (255, 255, 255), -1)
cv2.rectangle(mask2, (x2, y2), (x2 + w2, y2 + h2), (255, 255, 255), -1)
cv2.rectangle(mask3, (x3, y3), (x3 + w3, y3 + h3), (255, 255, 255), -1)
cv2.rectangle(mask4, (x4, y4), (x4 + w4, y4 + h4), (255, 255, 255), -1)
cv2.rectangle(mask5, (x5, y5), (x5 + w5, y5 + h5), (255, 255, 255), -1)
# 将掩膜应用到原始图像上
# masked_image1 = cv2.bitwise_and(image, mask1)
masked_image2 = cv2.bitwise_and(image, mask2)
masked_image3 = cv2.bitwise_and(image, mask3)
masked_image4 = cv2.bitwise_and(image, mask4)
masked_image5 = cv2.bitwise_and(image, mask5)
# 保存掩膜图像
# cv2.imwrite('D:/Paddle_ocr_project/picture/masked_image1.jpg', masked_image1)
cv2.imwrite('D:/Paddle_ocr_project/picture/masked_image2.jpg', masked_image2)
cv2.imwrite('D:/Paddle_ocr_project/picture/masked_image3.jpg', masked_image3)
cv2.imwrite('D:/Paddle_ocr_project/picture/masked_image4.jpg', masked_image4)
cv2.imwrite('D:/Paddle_ocr_project/picture/masked_image5.jpg', masked_image5)

# cv2.imshow('1', masked_image1)
cv2.imshow('2', masked_image2)
cv2.imshow('3', masked_image3)
cv2.imshow('4', masked_image4)
cv2.imshow('5', masked_image5)
cv2.waitKey(0)
cv2.destroyAllWindows()
ocr = PaddleOCR()
# result1 = ocr.ocr(masked_image1, cls=True, det=True)
result2 = ocr.ocr(masked_image2, cls=True, det=True)
result3 = ocr.ocr(masked_image3, cls=True, det=True)
result4 = ocr.ocr(masked_image4, cls=True, det=True)
result5 = ocr.ocr(masked_image5, cls=True, det=True)
# print(result3)
text_results = []


# for res in result2:
#    for line in res:
#        title = line[1][0]
#        text_results.append({
#            'pageName': title,
#        })


# for res in result1:
#   for line in res:
#      text = line[1][0]
##     text_results.append({
#         'text': text,
#         'box': box,
# })


def merge_text_results(results):
    # 创建一个变量用于保存上一行文字的box信息
    previous_box = None
    # 创建一个变量用于保存需要合并的文字
    merged_text = ''
    text_results = []

    for res in results:
        for line in res:
            text = line[1][0]
            box = line[0]

            # 检查previous_box是否已经有值（不是第一行）
            if previous_box is not None:
                # 计算当前行文字的y值与上一行的y值的差
                diff_y = abs(box[0][1] - previous_box[0][1])

                # 检查差值是否在10和25之间
                if diff_y < 35:
                    # 如果是，则将当前行的文字和merged_text合并
                    merged_text += '_' + text
                else:
                    # 如果不是，先将之前合并的文字添加到结果中
                    if merged_text:
                        text_results.append(merged_text)
                    # 然后开始新的合并
                    merged_text = text
            else:
                # 如果是第一行，开始新的合并
                merged_text = text

            # 更新previous_box为当前行的box
            previous_box = box

    # 将最后一行的合并结果也添加到结果中
    if merged_text:
        text_results.append(merged_text)

    return text_results


def merge_text_results_page(results):
    # 创建一个变量用于保存上一行文字的box信息
    previous_box = None
    # 创建一个变量用于保存需要合并的文字
    merged_text = ''
    text_results = []

    for res in results:
        for line in res:
            text = line[1][0]
            box = line[0]

            # 检查previous_box是否已经有值（不是第一行）
            if previous_box is not None:
                # 计算当前行文字的y值与上一行的y值的差
                diff_y = abs(box[0][1] - previous_box[0][1])

                # 检查差值是否在10和25之间
                if diff_y < 35:
                    # 如果是，则将当前行的文字和merged_text合并
                    merged_text += '_' + text
                else:
                    # 如果不是，先将之前合并的文字添加到结果中
                    if merged_text:
                        text_results.append({
                            'pageName': merged_text,
                        })
                    # 然后开始新的合并
                    merged_text = text
            else:
                # 如果是第一行，开始新的合并
                merged_text = text

            # 更新previous_box为当前行的box
            previous_box = box

    # 将最后一行的合并结果也添加到结果中
    if merged_text:
        text_results.append({
            'pageName': merged_text,
        })

    return text_results


result2_merged = merge_text_results_page(result2)
result3_merged = merge_text_results(result3)
result4_merged = merge_text_results(result4)
result5_merged = merge_text_results(result5)
print("r2", result2_merged)
print("r3", result3_merged)
print("r4", result4_merged)
print("r5", result5_merged)

# list = [result2_merged[0], {"pageContent": result3_merged + result4_merged + result5_merged}]
lst = {"pageName": "电阻屏_" + result2_merged[0].get("pageName"),
       "pageContent": result3_merged + result4_merged + result5_merged}
output_file = 'D:/Paddle_ocr_project/picture/ocr_results_new.json'
with open(output_file, 'w') as f:
    json.dump(lst, f)

json_file_path = output_file
with open(json_file_path, 'r') as file:
    json_data = file.read()
    file_name = os.path.basename(image_path)
    contentType = mimetypes.guess_type(image_path)
    image_path = os.path.realpath(image_path)
    file = {'file': (file_name, open(image_path, 'rb'), contentType)}
    msg = {'msg': json_data}
    print(msg)
response = requests.request('post', host, files=file, data=msg)
print(response.text)

# 根据服务器的响应进行相应的处理操作
if json.loads(response.text)['code'] == 200:
    print('JSON文件上传成功')
else:
    print('JSON文件上传失败')
