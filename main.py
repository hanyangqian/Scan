import cv2
import numpy as np
import argparse



# 设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

def img_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype='float32')

    # 按顺序找到对应0123对应左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算左上，右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # 计算输入的w和h的值
    widthA = np.sqrt((((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)))
    widthB = np.sqrt((((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)))
    heightB = np.sqrt((((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2)))
    maxHeight = max(int(heightA), int(heightB))

    # 变换后对应的坐标位置
    dst = np.array([
        [0,0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


img = cv2.imread('pos.jpg')
# 坐标相同变化
ratio = img.shape[0] / 500.0
orig = img.copy()

img = resize(orig, height = 500)

# 先拿到边缘信息
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# STEP 1 边缘检测结果
print('======STEP 1 边缘检测======')
img_show('edged', edged)

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]  # 拿到所有的轮廓
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[: 5]  # 按照矩形大小排序获取信息 拿到前五个大的 为了方便多个纸片的情况

# 遍历轮廓
for c in cnts:
    # 计算轮廓长度
    peri = cv2.arcLength(c, True)  # 计算长度
    # 计算一下轮廓近似 近似出来轮廓的矩形 True 表示封闭
    # c 输入点集
    # epsilon 表示从原始轮廓到近似轮廓的最大距离，是准确度参数
    # cv.approxPolyDP 函数返回的是多边形的顶点坐标数组
    approx = cv2.approxPolyDP(c, 0.02*peri, True)
    # 四个点的时候就拿出来
    if len(approx) == 4:
        screenCnt = approx
        break

# 展示结果
print("======STEP 2 获取轮廓======")
cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)
img_show('CNT', img)

# print(screenCnt)
# print(screenCnt.reshape(4, 2))

# 注意reshape
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 二值化
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imwrite("res.jpg", ref)
# 展示结果
print("======STEP 3 透视变换======")

#pos与res未响应问题
#imshow需要配合waitkey等
img_show("pos", resize(orig, height=650))
img_show("res", resize(ref, height=650))

