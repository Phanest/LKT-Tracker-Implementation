
import cv2
import numpy as np
from scipy import spatial
import helpers

class featurePoints:

    def __init__(self, camera, radius, NumFeatures=-1, isTest=False):
        img = None

        for i in range(0, 5):
            ret, img = camera.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.waitKey(1)

        kp = []
        if isTest:
            kp = [(200, 200)]
        else:
            kp = self.findPoints(img, radius, NumFeatures)

        self.radius = radius
        self.points = kp
        self.size = img.shape

    def findPoints(self, img, radius, number=-1):
        fast = cv2.FastFeatureDetector_create(20)
        kp = fast.detect(img, None)

        kp = [point.pt for point in kp]
        distMatrix = spatial.distance_matrix(kp, kp)

        keep = []
        exclude = []

        for i in range(0, len(kp)):
            if i in exclude:
                continue

            for j in range(i+1, len(kp)):
                if distMatrix[i, j] < radius:
                    exclude.append(j)

        indices = [i for i in range(0, len(kp)) if i not in exclude]
        for i in indices:
            if number == 0:
                break
            number -= 1

            x, y = int(kp[i][0]), int(kp[i][1])
            keep.append((x, y))

        return keep

    def export(self):
        return self.points

    def outliers(self):
        n, m = self.size
        A = []

        for i in range(0, len(self.points)):
            x, y = self.points[i]

            if x == 'm':
                continue
            if x < 0 or x >= m:
                A.append(('m', 'm'))
            elif y < 0 or y >= n:
                A.append(('m', 'm'))
            else:
                A.append((x, y))

        self.points = A

    def clean(self):
        A = []
        for i in range(0, len(self.points)):
            x, y = self.points[i]
            if x != 'm':
                A.append((x, y))
        self.points = A

    def illustrate(self, img, prevPoints, type='Point'):
        prevPoints = helpers.getPixels(prevPoints)
        points = helpers.getPixels(self.points)

        if type == 'Point':
            for pos in points:
                if pos[0] == 'm':
                    continue
                cv2.circle(img, pos, 3, (0, 0, 255), -1)
        else:
            for i in range(0, len(prevPoints)):
                pos = prevPoints[i]
                pos2 = points[i]

                if pos[0] == 'm' or pos2[0] == 'm':
                    continue

                cv2.arrowedLine(img, pos, pos2, (0, 0, 255), 3)

        return img

    def addPoint(self, i, x, y):
        n = len(self.points)
        # print('points before: ', self.points)
        if i < n:
            self.points.pop(i)
        self.points.insert(i, (x, y))
        # print('points after: ', self.points)

    def getPoint(self, i):
        return self.points[i]

    def removePoint(self, i):
        a = ('m', 'm')
        self.points[i] = a

    def __len__(self):
        return len(self.points)