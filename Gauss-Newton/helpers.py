import numpy as np
import cv2

class Telemetry:

    def __init__(self):
        self.A = {}

    def addMetric(self, name, value):
        a = (name, value)
        self.A[name] = value #todo

    def getVal(self, key):
        return self.A[key]

    def report(self):
        for key, value in self.A:
            print(key, ': ', value)

def cropAndPad(img, y, x, winSize):
    x, y = int(x), int(y)
    n, m = img.shape

    up = y - winSize
    down = y + winSize
    right = x + winSize
    left = x - winSize

    n = n - 1
    m = m - 1

    upPad = 0
    downPad = 0
    rightPad = 0
    leftPad = 0

    if up < 0:
        upPad = abs(up)
        up = 0
    if down > n:
        downPad = down - n - 1
        down = n
    if left < 0:
        leftPad = abs(left)
        left = 0
    if right > m:
        rightPad = right - m - 1
        right = m

    img = img[up:down, left:right]
    img = cv2.copyMakeBorder(img, upPad, downPad, leftPad, rightPad, cv2.BORDER_REPLICATE)

    return img

def Crop(img, y, x, winSize):
    n, m = img.shape

    up = y - winSize
    down = y + winSize
    right = x + winSize
    left = x - winSize

    if up < 0:
        up = 0
    if down > n:
        down = n
    if left < 0:
        left = 0
    if right > m:
        right = m

    return up, right, down, left

#type 0 for X, 1 for Y
def Gradient(img, type):
    n, m = img.shape
    Gradient = None

    if type == 0:
        GradientX = np.zeros((1, m), dtype=np.uint8) #One axis of the gradient in x direction
        for i in range(0, m):
            GradientX[0, i] = i
        GradientX = np.tile(GradientX, (n, 1)) #repeat axis n row times

        Gradient = GradientX
    else:
        GradientY = np.zeros((1, n), dtype=np.uint8)  # One axis of the gradient in y direction
        for i in range(0, n):
            GradientY[0, i] = i
        GradientY = np.tile(GradientY, (m, 1))  # repeat axis m row times
        GradientY = GradientY.T

        Gradient = GradientY

    return Gradient

def euclid(pos, pos2):
    from math import sqrt
    x, y = pos
    nx, ny = pos2

    a = (x-nx)**2
    b = (y-ny)**2

    a = a + b
    return sqrt(a)

def Warp(pos, p):
    x, y = pos

    nx = (1 + p[0])*x + p[2]*y + p[4]
    ny = p[1]*x + (1+p[3])*y + p[5]

    return nx, ny

def inBound(img, x, y, winSize):
    n, m = img.shape

    n = n - 1
    m = m - 1

    up = y - winSize
    right = x + winSize
    down = y + winSize
    left = x - winSize

    if left < 0 or up < 0:
        return False
    if right > m or down > n:
        return False

    return True

def getPixels(points):
    A = []

    for i in range(0, len(points)):
        x, y = points[i]

        if x == 'm':
            A.append(points[i])
            continue
        x, y = int(x), int(y)
        A.append((x, y))

    return A

def Show(img, frames=500):
    for i in range(0, frames):
        cv2.imshow('Test', img)
        cv2.waitKey(1)
    cv2.destroyWindow('Test')
