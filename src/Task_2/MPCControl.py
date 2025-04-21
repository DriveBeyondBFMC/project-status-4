import re
import osqp
import pyclothoids
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from typing import Tuple, Any
import configparser as conf
from tqdm import tqdm

class VehicleKinematic:
    def __init__(self, x = 0, y = 0, psi = 0, Lr = 26, Lf = 0):
        self.x = x
        self.y = y
        self.psi = psi # heading angle
        self.Lr = Lr
        self.Lf = Lf
        
        self.stateVect = np.array([self.x, self.y, self.psi])

    def updateRule(self, v, phi, psi):
        beta = np.arctan(self.Lr / (self.Lr + self.Lf) * np.tan(phi)) # Slip angle
        
        xdot = v * np.cos(psi + beta)
        ydot = v * np.sin(psi + beta)
        psidot = v * np.tan(beta) / self.Lr
        
        return xdot, ydot, psidot
    
    @classmethod
    def loadFile(cls, configPath):
        config = conf.ConfigParser()
        config.read(configPath)
        
        lf = float(config['VehicleDim']['lf'])
        lr = float(config['VehicleDim']['lr'])
        return cls(0, 0, 0, lr, lf)
    
    def updateState(self, v, phi, dt = 0.01):
        x1 = [v, phi, self.psi]
        f1 = self.updateRule(*x1)
        
        x2 = [v, phi, self.psi + f1[2] * dt / 2]
        f2 = self.updateRule(*x2)   
        
        x3 = [v, phi, self.psi + f2[2] * dt / 2]
        f3 = self.updateRule(*x3)
        
        x4 = [v, phi, self.psi + f3[2] * dt]
        f4 = self.updateRule(*x4)
        
        self.x += dt / 6 * (f1[0] + 2*f2[0] + 2*f3[0] + f4[0])
        self.y += dt / 6 * (f1[1] + 2*f2[1] + 2*f3[1] + f4[1])
        self.psi += dt / 6 * (f1[2] + 2*f2[2] + 2*f3[2] + f4[2])
        self.stateVect = np.array([self.x, self.y, self.psi])

    def stateSpace(self, v, psiStab, timestep):
        A = np.eye(3)
        # Term betaStab exist but it is now zero
        B = lambda psiStab, v: np.array([[-v * np.sin(psiStab)],
                                        [v * np.cos(psiStab)],
                                        [(v / self.Lr) * np.cos(0)]]) * timestep
        # Constant vector
        K = lambda psiStab, v: np.array([[np.cos(psiStab)],
                                        [np.sin(psiStab)],
                                        [np.cos(0) / self.Lr]]) * v * timestep

        return A, B(psiStab, v), K(psiStab, v)

class PathTracker:
    def __init__(self, trackingRad, path):
        self.trackingRad = trackingRad
        self.posRel2Path = 0
        self.pathLength = len(path[0])
        
    def tracking(self, distance):
        """
        Update the relative path index (posRel2Path) based on the vehicle's current distance
        to the path. Ensures the tracker always moves forward and handles large deviations.
        """
        # Search in a forward-looking window (optional tuning)
        lookahead = 50
        search_start = self.posRel2Path
        search_end = min(search_start + lookahead, self.pathLength)

        # Get sub-distance and find closest in this window
        local_dist = distance[search_start:search_end]
        closest_local_idx = np.argmin(local_dist) + search_start
        closest_global_idx = np.argmin(distance)

        # Threshold for forcing a reset (vehicle far from path)
        far_threshold = 3 * self.trackingRad

        if distance[closest_global_idx] > far_threshold:
            # If too far from entire path, jump to global closest
            self.posRel2Path = closest_global_idx
        else:
            # Otherwise, update forward if the closest is ahead
            if closest_local_idx > self.posRel2Path:
                self.posRel2Path = closest_local_idx

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
        offsetPoints = [];
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
        s = np.concat(([0], np.cumsum(ds)))

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
        

    def distance2Path(self, position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(position - self.definedPath[:2], axis = 0)
    
    def distance2MajorPoint(self, position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(position - self.majorPoint[:2], axis = 0)

    
vehicle2Path = lambda x, y, path: np.linalg.norm([x - path[0], y - path[1]], axis=0)


def blockBidiagMat(As, I):
    N = As.shape[0]
    bidiag = np.kron(np.eye(N), I)
    blockSz = I.shape[0]
        
    diagA = sp.block_diag([sp.csc_matrix(-A) for A in As]).toarray()
    
    return np.pad(diagA, ((blockSz, 0), (0, 0)))[:-blockSz] + bidiag


        
if __name__ == "__main__":
    # path definition
    xPath = [0, 300, 390, 390]
    yPath = [0, 0,   -90,  -200]
    theta = [0, 0, -np.pi / 2, -np.pi / 2]  
        
    pathProcessor = PathHandler.clothoidsGenFile('./plannedCoor.txt')

    # vehicle definition
    lr = 26
    lf = 0
    v = 20

    phi = 0 * np.pi / 180
    dt = 0.075
    timer = 150

    # ======================== INITIALIZE MPC ====================================    
    
    stateTs = 0.03
    N = 10 # Horizon length
    terminalGain = 200
    initStateGain = 1
    stateGain = 1
    actuationGain = 10
    deltaActuationGain = 100000

    # ====================== Unchanging MPC matrix parameters ====================
    # Cost Function weights
    n = 3; m = 1
    Q = np.eye(n) * stateGain
    Qr = np.eye(n) * initStateGain
    Qf = np.eye(n) * terminalGain
    Q[-1, -1] = 0; Qr[-1, -1] = 0; Qf[-1, -1] = 0 # No need to penalize vehicle orientation
    R = np.eye(m) * actuationGain
    D = np.kron(np.eye(N - 1, N, k=1) - np.eye(N - 1, N), np.eye(m))
    Rdelta = np.eye(m) * deltaActuationGain

    # Pre initialize unchanging quadratic cost
    Qblock = 2 * sp.block_diag([sp.csc_matrix(Qr)] + [sp.csc_matrix(Q) for _ in range(N-2)] + [sp.csc_matrix(Qf)]) # Weight of state
    Rblock = 2 * sp.block_diag([sp.csc_matrix(R) for _ in range(N)]) # Weight of control actuation
    RdeltaBlock = 2 * sp.csc_matrix(D.T @ np.kron(np.eye((N - 1)), Rdelta) @ D) # Weight of control actuation difference
    H = sp.block_diag([Qblock, Rblock + RdeltaBlock]).tocsc() # Hessian matrix (Combination of all 3 Weights)

    # Initialize inequality equation
    limitSteering = 30 * np.pi / 180
    limitSlipAngle = np.atan((lr / (lr + lf)) * np.tan(limitSteering))

    xMin = np.array([-np.inf, -np.inf, -np.inf])
    xMax = np.array([np.inf, np.inf, np.inf])
    uMin = np.array([-limitSlipAngle])
    uMax = np.array([limitSlipAngle])

    Xmin = np.tile(xMin, N); Xmax = np.tile(xMax, N)
    Umin = np.tile(uMin, N); Umax = np.tile(uMax, N)

    Aineq = sp.eye((n + m) * N, format = 'csc')
    Lineq = np.r_[Xmin, Umin] 
    Uineq = np.r_[Xmax, Umax]
    
    
    prob = osqp.OSQP()
    inputs = []
    
    # ======================== INITIALIZE MPC ====================================    
    
    vehicle = VehicleKinematic.loadFile("./controller.conf")
    vehicle.x = pathProcessor.definedPath[0, 0]
    vehicle.y = pathProcessor.definedPath[1, 0]
    tracker = PathTracker(trackingRad = 50, path = pathProcessor.definedPath)
        

    vehicleCoor, predState = [], []
    for _ in tqdm(range(int(timer / dt) )):
        vehicle.updateState(v, phi, dt)
        x0 = vehicle.stateVect
        
        distance = vehicle2Path(vehicle.x, vehicle.y, pathProcessor.definedPath)
        tracker.tracking(distance)
        
        refState = pathProcessor.generateReference(v, stateTs, tracker.posRel2Path, N)
        stateCoefficient = [vehicle.stateSpace(v, psiStab, stateTs) for psiStab in refState[:, -1]]
        As, Bs, Ks = zip(*stateCoefficient)
        AsArray = np.array(As); BsArray = np.array(Bs); CsArray = np.array(Ks)

        # Linear cost calculation
        fX = (-2 * Q @ refState.T).T.flatten()
        fX[-n:] *= terminalGain
        fU = np.zeros(m * N)
        f = np.r_[fX, fU]

        # Equality equation construction
        Aeqx = blockBidiagMat(AsArray, np.eye(n))
        Aequ = sp.block_diag([-B for B in Bs]).toarray()
        Aeq = np.c_[Aeqx, Aequ]
        beq = np.r_[As[0] @ x0 + Ks[0].T[0], *[Ks[i].T[0] for i in range(1, N)]]
        
        # The complete equation (including equality and inequality)
        Atotal = sp.vstack([Aeq, Aineq], format = 'csc')
        Ltotal = np.concat([beq, Lineq])
        Utotal = np.concat([beq, Uineq])
        
        prob.setup(P = H, q = f, A = Atotal, l = Ltotal, u = Utotal, verbose = False)
        res = prob.solve()
        z = res.x

        inputActuation = z[n*N]
        # print(z[n * N: ])
        phi = np.atan(((lf + lr) / lr) * np.tan(inputActuation))
        inputs += [phi * 180 / np.pi]
        
        
        vehicleCoor.append([vehicle.x, vehicle.y, vehicle.psi])
        predState.append(z[:n * N].reshape((N, n)))


    vehicleCoor = np.array(vehicleCoor) 
    predState = np.array(predState)

    fig, ax = plt.subplots(2, 1, figsize=(16, 10))

    
    ax[1].plot(inputs)
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Angle (deg)')
    ax[1].set_title("Input actuation")
    ax[1].grid()
    
    ax[0].plot(pathProcessor.definedPath[0], pathProcessor.definedPath[1], label = "Clothoid Path")
    ax[0].set_title("Vehicle Path")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_xlim(min(np.min(pathProcessor.definedPath[0]), np.min(vehicleCoor[:, 0])) - 20, 
                   max(np.max(pathProcessor.definedPath[0]), np.max(vehicleCoor[:, 0])) + 20)
    ax[0].set_ylim(min(np.min(pathProcessor.definedPath[1]), np.min(vehicleCoor[:, 1])) - 20, 
                   max(np.max(pathProcessor.definedPath[1]), np.max(vehicleCoor[:, 1])) + 20)
    ax[0].legend()
    ax[0].set_aspect('equal')
    ax[0].grid()

    vehicle_line, = ax[0].plot([], [], 'orange', lw=2, label="Vehicle Path")
    predicted_line, = ax[0].plot([], [], 'b--', lw=1.5, label="Predicted Trajectory")
    actuation_dot = Line2D([], [], color='red', marker='o', markersize=8)

    # Rectangle (vehicle body)
    vehicle_length = lr + lf
    vehicle_width = 10
    vehicle_rect = patches.Rectangle(
        (-vehicle_length / 2, -vehicle_width / 2),
        vehicle_length,
        vehicle_width,
        edgecolor='red',
        facecolor='none',
        linewidth=2
    )
    ax[0].add_patch(vehicle_rect)
    ax[1].add_line(actuation_dot)

    # === Initialization function ===
    def init():
        vehicle_line.set_data([], [])
        predicted_line.set_data([], [])
        vehicle_rect.set_xy((-vehicle_length / 2, -vehicle_width / 2))
        actuation_dot.set_data([], [])
        return vehicle_line, vehicle_rect, predicted_line, actuation_dot

    # === Animation update function ===
    def update(frame):
        # Vehicle path
        x_data = vehicleCoor[:frame, 0]
        y_data = vehicleCoor[:frame, 1]
        psi = vehicleCoor[frame - 1, 2]
        x = vehicleCoor[frame - 1, 0]
        y = vehicleCoor[frame - 1, 1]

        vehicle_line.set_data(x_data, y_data)

        # Update rectangle transformation
        trans = patches.transforms.Affine2D().rotate_around(0, 0, psi).translate(x, y) + ax[0].transData
        vehicle_rect.set_transform(trans)

        # Predicted trajectory
        if frame < len(predState):
            pred = predState[frame]  # shape: (N, 3)
            predicted_line.set_data(pred[:, 0], pred[:, 1])
        else:
            predicted_line.set_data([], [])

        # Update actuation dot
        if frame < len(inputs):
            actuation_dot.set_data([frame], [inputs[frame]])
        else:
            actuation_dot.set_data([], [])

        return vehicle_line, vehicle_rect, predicted_line, actuation_dot

        
    # === Run the animation ===
    ani = animation.FuncAnimation(
        fig, update,
        frames=len(vehicleCoor),
        init_func=init,
        blit=True,
        interval=5,
        repeat=True
    )

    plt.tight_layout()
    plt.show()