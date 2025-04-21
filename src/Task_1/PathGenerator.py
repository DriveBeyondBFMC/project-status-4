import re
import numpy as np
import pyclothoids
from scipy.interpolate import interp1d

class PathHandler:
    def __init__(self, definedPath: np.ndarray, majorPoint: np.ndarray, cameraSwitch: np.ndarray = None):
        self.definedPath = definedPath
        self.cameraSwitch = cameraSwitch
        self.majorPoint = majorPoint    
        self.pathInterpolation

        self.definedPathCopy = definedPath.copy()
    
    @classmethod
    def clothoidsGenManual(cls, x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> "PathHandler":
        return cls(cls.__generator__(x, y, theta))
    
    @classmethod
    def clothoidsGenFile(cls, pathName: str) -> "PathHandler":
        dataArr = {}
        with open(pathName, "r") as file:
            for line in file:
                key, values = line.split(":")
                # Remove brackets and commas, then split into a list
                cleaned_values = re.sub(r"[\[\],]", "", values).strip()
                values_list = np.array([float(v) for v in cleaned_values.split()])
                dataArr[key.strip()] = values_list

        # Accessing the arrays
        yPath = dataArr["Array_y"] 
        xPath = dataArr["Array_x"] 
        theta = np.pi * 2 - (dataArr["Array_z"] + np.pi / 2)
        cameraSwitch = dataArr["Array_t"]
        
        mask = np.insert((np.diff(xPath) != 0) | (np.diff(yPath) != 0), 0, True)

        xPath = xPath[mask]
        yPath = yPath[mask]
        theta = theta[mask]
        cameraSwitch = cameraSwitch[mask]
        return cls(cls.__generator__(xPath, yPath, theta), np.r_[[xPath], [yPath], [theta]], cameraSwitch)
    
    @staticmethod
    def __generator__(x, y, theta):
        xVals, yVals, headings = [], [], []
        ds = 1e-4

        for i in range(min(len(theta), len(x), len(y)) - 1):
            clothoid = pyclothoids.Clothoid.G1Hermite(x[i], y[i], theta[i], x[i+1], y[i+1], theta[i+1])
            pointDist = np.linalg.norm([x[i+1] - x[i], y[i+1] - y[i]])

            sVals = np.linspace(0, clothoid.length - ds, int(pointDist))
            
            for s in sVals:
                px = clothoid.X(s)
                py = clothoid.Y(s)
                dx = clothoid.X(s + ds) - px
                dy = clothoid.Y(s + ds) - py
                angle = np.arctan2(dy, dx)

                xVals.append(px)
                yVals.append(py)
                headings.append(angle)
        return np.array([xVals, yVals, headings])

    def generateOffsetPath(self, offsetDist: float):
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                        [np.sin(base[2]), np.cos(base[2]), base[1]],
                                        [0, 0, 1]])
        offsetPoints = []; self.offsetDist = offsetDist
        for majorPoint in self.majorPoint.T:
            
            offsetMajorPoint = (rotmat(majorPoint) @ np.array([[0, offsetDist, 1]]).T).T[0]
            offsetMajorPoint[2] = majorPoint[2]
            offsetPoints += [offsetMajorPoint]
        offsetPoints = np.array(offsetPoints).T
        
        self.offsetPath = self.__generator__(*offsetPoints)
    
    @property
    def pathInterpolation(self):
        pathDiff = np.diff(self.definedPath, axis = 1)
        ds = np.sqrt(np.sum(pathDiff ** 2, axis = 0))
        s = np.concatenate(([0], np.cumsum(ds)))

        xInterp = interp1d(s, self.definedPath[0], kind = 'linear')
        yInterp = interp1d(s, self.definedPath[1], kind = 'linear')
        psiInterp = interp1d(s, self.definedPath[2], kind = 'linear')
        
        self.pathInterp = (xInterp, yInterp, psiInterp)
        self.s = s

    def generateReference(self, vRef: float, dt: float, startIdx: int, N: int) -> np.ndarray:
        refTraj = []
        for i in range(N):
            s_i = min(self.s[startIdx] + vRef * i * dt, self.s[-1])
            refTraj.append([
                self.pathInterp[0](s_i),
                self.pathInterp[1](s_i),
                self.pathInterp[2](s_i),
            ])
            
        return np.array(refTraj)
    
    def pathAvoidance(self, positionIdx: np.ndarray, offsetOnPath: float, offsetDist: np.ndarray, baseLength: float):
        startIdx = positionIdx
        
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                         [np.sin(base[2]), np.cos(base[2]), base[1]],
                                         [0, 0, 1]])

        # ========  Define 4 points for avoidance path =======
        start = np.array([self.pathInterp[i](self.s[startIdx]) for i in range(3)])
        
        offset1 = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath) for i in range(3)])

        offset2Base = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist[0]) for i in range(3)])
        offset2 = (rotmat(offset2Base) @ np.array([[0, offsetDist[1], 1]]).T).T[0]
        
        
        offset3Base = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist[0] + baseLength) for i in range(3)])
        offset3 = (rotmat(offset3Base) @ np.array([[0, offsetDist[1], 1]]).T).T[0]

        end = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist[0] * 2 + baseLength) for i in range(3)])
        # ========  Define 4 points for avoidance path =======

        targetS = self.s[startIdx] + offsetDist[0] * 2 + baseLength + offsetOnPath
        endIdx = np.searchsorted(self.s, targetS)
        
        offset2[2] = offset2Base[2]
        offset3[2] = offset3Base[2]

        offsetPath = self.__generator__(
            *zip(*[start, offset1, offset2, offset3, end])
        )

        self.definedPath = np.concatenate(
            (self.definedPath[:, :startIdx], offsetPath, self.definedPath[:, endIdx + 20:]),
            axis = 1
        ) # add new path

        return offsetPath

    def entryRamp(self, positionIdx: int, 
                  offsetPositionIdx: int, 
                  offsetOnPath: float, 
                  offsetDist: float, 
                  possibleEnd: float = 0, 
                  epsilon: float = 1) -> np.ndarray:
        startIdx = positionIdx
        
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                         [np.sin(base[2]), np.cos(base[2]), base[1]],
                                         [0, 0, 1]])

        start = np.array([self.pathInterp[i](self.s[startIdx]) for i in range(3)])
        
        mid1 = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath) for i in range(3)])

        mid2Offset = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist) for i in range(3)])
        mid2 = (rotmat(mid2Offset) @ np.array([[0, self.offsetDist, 1]]).T).T[0]
        mid2[2] = mid2Offset[2]
        
        endOffset = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist + possibleEnd) for i in range(3)])
        end = (rotmat(endOffset) @ np.array([[0, self.offsetDist, 1]]).T).T[0]
        end[2] = endOffset[2]

        def lastIndex(idxState):
            dist = np.linalg.norm(idxState[:2][..., None] - self.offsetPath[:2], axis = 0)
            minIndices = np.where(dist <= dist.min() + epsilon)[0]
            if len(minIndices) > 1:
                lastIdx = min(minIndices, key=lambda x: abs(x - offsetPositionIdx))
            else:
                lastIdx = minIndices[0]
            
            return lastIdx
        
        lastIdxRamp = lastIndex(mid2)
        lastIdxPossibleLength = lastIndex(end)
        
        
        enterRamp = self.__generator__(*zip(*[start, mid1, mid2]))
        self.offsetPath = np.concatenate([self.offsetPath[:, :offsetPositionIdx], enterRamp, self.offsetPath[:, lastIdxRamp:]], axis = 1)

        return lastIdxPossibleLength    
    
    def exitRamp(self, positionIdx: int, offsetOnPath: float, offsetDist: float) -> np.ndarray:
        startIdx = positionIdx
        
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                         [np.sin(base[2]), np.cos(base[2]), base[1]],
                                         [0, 0, 1]])
        
        startOffset = np.array([self.pathInterp[i](self.s[startIdx]) for i in range(3)])
        start = (rotmat(startOffset) @ np.array([[0, self.offsetDist, 1]]).T).T[0]
        start[2] = startOffset[2]

        mid = np.array([self.pathInterp[i](self.s[startIdx] + offsetDist) for i in range(3)])
        
        end = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist) for i in range(3)])
        
        targetS = self.s[startIdx] + offsetDist + offsetOnPath
        endIdx = np.searchsorted(self.s, targetS)
        
        exitRamp = self.__generator__(*zip(*[start, mid, end]))
        self.definedPath = np.concatenate([self.definedPath[:, :startIdx], exitRamp, self.definedPath[:, endIdx:]], axis = 1)
        self.pathInterpolation

        return endIdx
        
    def distance2Path(self, position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(position - self.definedPath[:2], axis = 0)
    
    def distance2MajorPoint(self, position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(position - self.majorPoint[:2], axis = 0)