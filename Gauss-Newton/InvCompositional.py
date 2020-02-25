#Kanade-Lucas-Tomasi

import cv2
import numpy as np
import math
import helpers
from FeaturePoints import featurePoints

def SpatialGradient(Ix, Iy, x, y, winSize):
    G = np.matrix('0 0; 0 0')

    n, m = Ix.shape

    iX = x - winSize
    fX = x + winSize
    iY = y - winSize
    fY = y + winSize

    if iX < 0: iX = 0
    if fX > m: fX = m-1
    if iY < 0: iY = 0
    if fY > n: fY = n-1

    for i in range(iY, fY):
        for j in range(iX, fX):
            a = Ix[i, j]
            b = Iy[i, j]
            try : G[0, 0] += a*a
            except: return Ix, Iy
            G[0, 1] += a*b
            G[1, 0] += a*b
            G[1, 1] += b*b

    return G

def imageMismatch(diffI, Ix, Iy, x, y, winSize):
    b = np.matrix('0 ; 0')

    n, m = Ix.shape

    iX = x - winSize
    fX = x + winSize
    iY = y - winSize
    fY = y + winSize

    if iX < 0: iX = 0
    if fX > m: fX = m-1
    if iY < 0: iY = 0
    if fY > n: fY = n-1

    for i in range(iY, fY):
        for j in range(iX, fX):
            a = Ix[i, j]*diffI
            c = Iy[i, j]*diffI
            b[0, 0] += a
            b[1, 0] += c

    return b

def findCoordinates(dst):
    coordinates = []
    n, m = dst.shape
    max = dst.max()

    for i in range(0, n):
        for j in range(0, m):
            if dst[i, j] > 0.01 * max:
                coordinates.append([i, j])

    return coordinates

#type - function or points
#features - corner detection function or list of points
def trackFeatures(camera, type='function', features=cv2.cornerHarris, refreshRate=120, treshold=0.5, steps=-1, winSize = 10):

    dst = []

    count = 0
    frame = refreshRate
    prevImage = None
    coordinates = None
    oldCoordinates = None

    if type != 'function':
        coordinates = features

    #Taking just one frame sometimes gives artifacts so we take longer to take a frame
    for i in range(0, 5):
        ret, prevImage = capture.read()
        prevImage = cv2.cvtColor(prevImage, cv2.COLOR_BGR2GRAY)

    while True:
        ret, Image = capture.read()
        newImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

        #Refresh corners after a while
        if type == 'function' and frame >= refreshRate:
            gray = np.float32(newImage)
            dst = cv2.cornerHarris(gray, 2, 3, 0.04)
            coordinates = findCoordinates(dst)
            frame = 0

        for (x, y) in coordinates:
            Image[x, y] = [0, 0, 255]

        Pyr = []
        PyrJ = []

        Pyr.append(prevImage)
        PyrJ.append(newImage)

        for i in range(1, 4):
            Pyr.append(cv2.pyrDown(Pyr[i-1]))
            PyrJ.append(cv2.pyrDown(PyrJ[i-1]))
        Pyr.reverse()
        PyrJ.reverse()

        #Set guesses to 0
        #We keep guesses for each coordinate here
        guesses = []
        for i in range(0, len(coordinates)):
            guesses.append(np.array([0, 0]))

        for i in range(0, 4):
            j = 4-(i+1)

            img = Pyr[i]
            imgJ = PyrJ[i]

            Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # ksize
            Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # ksize

            for index, coordinate in enumerate(coordinates):
                x, y = coordinate
                x = math.ceil(x/(2**j))
                y = math.ceil(y/(2**j))

                G = SpatialGradient(Ix, Iy, x, y, winSize)
                try: Ginv = np.linalg.inv(G)
                except:
                    print(x, y)
                    print(G)
                    print(Ix, Iy)
                    return G, x, y, Pyr

                g = guesses[index]
                v = np.matrix('0 ; 0') #x, y
                n = np.matrix('100 ; 100')
                m, l = img.shape

                for step in range(0, steps):
                    if (n[0, 0] <= treshold and n[1, 0] <= treshold):
                        break
                    a = math.ceil(y+g[0]+ v[0, 0]) #y
                    b = math.ceil(x+g[1]+v[1, 0]) #x

                    if a < 0 or a >= m:
                        y = m
                        break
                    if b < 0 or b >= l:
                        x = l
                        break
                    diffI = img[y, x] - imgJ[a, b]
                    b = imageMismatch(diffI, Ix, Iy, x, y, winSize)
                    n = Ginv*b
                    v = v + n

                d = v
                if i < 3:
                    g.flat[0] = 2*(g[0] + d[0, 0]) #x
                    g.flat[1] = 2*(g[1] + d[1, 0]) #y
                else:
                    g.flat[0] = g[0] + d[0, 0]
                    g.flat[1] = g[1] + d[1, 0]

                guesses[index] = g #L-1

        oldCoordinates = coordinates.copy()

        delete = []
        for i in range(0, len(coordinates)):
            g = guesses[i]
            coordinate = coordinates[i]
            n, m = prevImage.shape
            x = math.ceil(coordinate[1] + g[0]) #coordinate 1
            y = math.ceil(coordinate[0] + g[1])
            if x < 0 or x >= m or y < 0 or y >= n:
                delete.add(i)
            else:
                coordinates[i] = [x, y]
        for i in delete:
            coordinates.pop(i)

        prevImage = newImage
        frame += 1

        cv2.imshow('Display', Image)
        cv2.waitKey(1)

    #if function take features
    #for i in 1:4 ?parameter
    #scale points by u/2^l=i
    #Find derivatives [Ix, Iy]T Ix means on whole image
    #Find gradient, find size of windows
    #Set v = []
    #While steps or treshold
    #Find difference between prevImage and image
    #Find b, and n
    #Guess next v

    #d = final V
    #g - next guesses

    #Find d
    #Find next point
    #Make line from previous point to the next one

    pass

def validPoints(n, m, points):
    p = []
    for x, y in points:
        if x < 0 or x >= m:
            continue
        if y < 0 or y >= n:
            continue
        p.append([x, y])
    return p

def getTW(StDesc, i, j, winSize):
    TW = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float32)

    TW[0, 0] = StDesc[i, j]
    TW[1, 0] = StDesc[i, j+winSize]
    TW[2, 0] = StDesc[i, j+2*winSize]
    TW[3, 0] = StDesc[i, j+3*winSize]
    TW[4, 0] = StDesc[i, j+4*winSize]
    TW[5, 0] = StDesc[i, j+5*winSize]

    return TW

def findHessian(StDesc, winSize):
    B = np.zeros((6, 6), np.float32)

    for j in range(6):
        for i in range(6):
            B[j, i] = np.sum(np.multiply(StDesc[0:winSize, i*winSize:(i+1)*winSize], StDesc[0:winSize, j*winSize:(j+1)*winSize]))
    #i, j ?

    return B

def getValue(img, x, y):
    n, m = img.shape

    if y < 0 or y >= n:
        return 255
    if x < 0 or x >= m:
        return 255

    return img[y, x]

def Warp(point, p):
    x, y = point

    a = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    p5 = p[4]
    p6 = p[5]
    p = a

    xUpdate = (1 + p)*x + p3*y + p5
    yUpdate = p2*x + (1+p4)*y + p6

    xUpdate = math.ceil(xUpdate)
    yUpdate = math.ceil(yUpdate)

    return (xUpdate, yUpdate)

def findUpdates(stDesc, diffI, winSize):
    A = stDesc.T
    sum = np.array([0, 0, 0, 0, 0, 0], np.float32)
    for i in range(0, 6):
        B = A[i*winSize:(i+1)*winSize, 0:winSize]
        s = B*diffI
        sum[i] = np.sum(s)
    return sum.T

def findNewPoints(points, p):
    newPoints = []

    a = p[0, 0]
    p2 = p[1, 0]
    p3 = p[2, 0]
    p4 = p[3, 0]
    p5 = p[4, 0]
    p6 = p[5, 0]
    p = a

    for x, y in points:
        xUpdate = (1 + p)*x + p3*y + p5
        yUpdate = p2*x + (1+p4)*y + p6
        newPoints.append([xUpdate, yUpdate])

    return newPoints

def showPoints(points, winSize):

    x, y = points

    points = np.array([[x, y], [x + winSize, y], [x + winSize, y + winSize], [x, y + winSize]])
    points = points.reshape((-1, 1, 2))

    return points

def findSteepestDescent(Ix, Iy):
    #Jacobian used
    # [x 0 y 0 1 0]
    # [0 x 0 y 0 1]
    n, m = Ix.shape #winSize
    winSize = n
    x = np.zeros((winSize, winSize), np.uint8)
    for i in range(0, winSize):
        for j in range(0, winSize):
            x[i, j] = j
    y = x.T

    x1 = Ix*x
    y1 = Ix*y
    ones1 = Ix

    x2 = Iy * x
    y2 = Iy * y
    ones2 = Iy

    J = np.hstack((x1, x2, y1, y2, ones1, ones2))
    # cv2.imshow('T', J)
    return J

def outOfBound(img, x, y, winSize):
    n, m = img.shape

    if x + winSize >= m:
        return True
    if y + winSize >= n:
        return True
    if x < 0:
        return True
    if y < 0:
        return True

    return False

def findPoints(capture, amount):
    points = []
    for i in range(0, 5):
        ret, img = capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        n, m = img.shape

        dst = cv2.cornerHarris(img, 2, 3, 0.04)
        dst = dst>0.01*dst.max()

        for i in range(0, len(dst)):
            for j in range(0, len(dst[0])):
                if amount == 0:
                    break
                if dst[i][j] == True:
                    if len(points) == 0:
                        x, y = points[len(points)-1]
                        points.append((j, i))
                        amount -= 1
                        continue

                    x, y = points[len(points)-1]
                    if j-50 < x < j+50 and i-50 < y < i+50:
                        points.append((j, i))
                        amount -= 1

        cv2.waitKey(1)

    return points

def invWrap(dp):
    p1 = dp[0]
    p2 = dp[1]
    p3 = dp[2]
    p4 = dp[3]
    p5 = dp[4]
    p6 = dp[5]

    d = (1 + p1)*(1 + p4) - p2*p3
    d = 1/d

    v = np.array([[-p1 - p1*p4 + p2*p3],
                  [-p2],
                  [-p3],
                  [-p4 - p1*p4 + p2*p3],
                  [-p5 - p4*p5 + p3*p6],
                  [-p6 - p1*p6 + p2*p5] ], dtype=np.float32)

    dp = d*v

    return dp.T[0]

def UpdateP(p, dp):
    dp = invWrap(dp)
    
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    p5 = p[4]
    p6 = p[5]

    dp1 = dp[0]
    dp2 = dp[1]
    dp3 = dp[2]
    dp4 = dp[3]
    dp5 = dp[4]
    dp6 = dp[5]

    w = np.array([[p1 + dp1 + p1*dp1 + p3*dp2],
                  [p2 + dp2 + p2*dp1 + p4*dp2],
                  [p3 + dp3 + p1*dp3 + p3*dp4],
                  [p4 + dp4 + p2*dp3 + p4*dp4],
                  [p5 + dp5 + p1*dp5 + p3*dp6],
                  [p6 + dp6 + p2*dp5 + p4*dp6]], dtype=np.float32)

    return w.T[0]

def KLTracker(capture, features, treshold=10, winSize=100):

    prevImage = None

    # Taking just one frame sometimes gives artifacts so we take longer to take a frame
    for i in range(0, 5):
        ret, prevImage = capture.read()
        prevImage = cv2.cvtColor(prevImage, cv2.COLOR_BGR2GRAY)
        cv2.waitKey(1)

    while True:
        ret, newImg = capture.read()
        grey = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)

        if ret == False:
            break

        prevPoints = features.export()  # removed points show as ('m', 'm')

        #Previous image is the template
        Tx = cv2.Sobel(prevImage, cv2.CV_64F, 1, 0, ksize=5)  # ksize
        Ty = cv2.Sobel(prevImage, cv2.CV_64F, 0, 1, ksize=5)  # ksize

        for i in range(0, len(features)):
            p = np.array([0, 0, 0, 0, 0, 0])
            dp = np.array([100, 100, 0, 0, 0, 0])

            x, y = features.getPoint(i)
            M = False

            if outOfBound(Tx, x, y, winSize):
                features.removePoint(i)
                continue

            dX = Tx[y:y + winSize, x:x + winSize]
            dY = Ty[y:y + winSize, x:x + winSize]

            # cv2.imshow('Previous Image - Template', prevImage[y:y + winSize, x:x + winSize])


            stDesc = findSteepestDescent(dX, dY)
            H = findHessian(stDesc, winSize)
            try:
                invH = np.linalg.inv(H)
                # cv2.imshow('H', invH)
            except:
                features.removePoint(i)
                continue

            while math.sqrt(dp[0]*dp[0] + dp[1]*dp[1]) > treshold:

                nx, ny = Warp((x, y), p)
                n, m = grey.shape
                
                if outOfBound(grey, nx, ny, winSize):
                    features.removePoint(i)
                    M = True
                    break

                try:
                    gr = grey[ny:ny + winSize, nx:nx + winSize]
                except:
                    print('Outside bounds')
                    M = True
                    break


                diffI = prevImage[y:y + winSize, x:x + winSize] - gr
                pUpdate = findUpdates(stDesc, diffI, winSize)
                dp = invH @ pUpdate
                p = UpdateP(p, dp)
                #p = p + dp

            if not M:
                newPoints = Warp((x, y), p)
                nx, ny = newPoints
                features.addPoint(i, nx, ny)

        features.outliers()
        newImg = features.illustrate(newImg, prevPoints, type='Point')

        features.clean()
        prevImage = grey

        cv2.imshow('Display', newImg)#[...,::-1,:])
        cv2.waitKey(1)

source = '../A.mp4' #Video stream or file
capture = cv2.VideoCapture(source)

if capture is None:
    print('None')

winSize = 50

features = featurePoints(capture, winSize, 20)
KLTracker(capture, features, treshold=1, winSize=50)