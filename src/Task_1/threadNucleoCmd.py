# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import numpy as np
import threading
import serial
import time
import socket
import psutil
import logging

from typing import Tuple

from src.templates.threadwithstop import ThreadWithStop
from src.utils.controllerUtils.controller import *
from src.utils.controllerUtils.PathGenerator import *
from src.utils.controllerUtils.obstacleAndSensor import subdivide_obstacle, detect_objects_with_dimensions
from src.utils.controllerUtils.SteeringFusion import fuser

import time


class threadNucleoCmd(ThreadWithStop):
    """Thread which will handle command functionalities.\n
    Args:
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger: logging, debugger):
        super(threadNucleoCmd, self).__init__()
        self.daemon = False
        self.queuesList = queuesList
        self.logger = logger.getLogger("threadNucleoCmd"); self.logger.setLevel(logging.DEBUG)
        self.debugger = debugger

        self.stopEvent = threading.Event() 
        

        # SETUP CONNECTION TO STM
        self.serial_port = '/dev/ttyACM0'
        self.baud_rate = 115200
        self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
        responseThread = threading.Thread(target = self.readResponse, args = (), daemon = True)
        responseThread.start()

        # SETUP CONNECTION FOR SOCKET
        self.setupConnection()

        # AVAILABLE COMMANDS
        self.commands = {
            '0': '#kl:30;;\r\n',
            '1': '#battery:0;;\r\n',
            '2': '#instant:0;;\r\n',
            '3': '#imu:0;;\r\n',
            '4': '#resourceMonitor:0;;\r\n',
            '5': '#speed:{0};;\r\n',
            '6': '#steer:{0};;\r\n',
            '7': '#vcd:200;0;121;\r\n',
            '8': '#vehicledynamic:{0};;\r\n',
            '9': '#vehicledynamic:1;;\r\n'
        }

        # INIT ALL VARIABLES
        # Receive/send from/to nucleo
        self.speedParam = 0
        self.actualSpeed = 0
        self.steerParam = 0
        self.states, self.initDynamic, self.statesUWB, self.tempState = [[0, 0, 0] for _ in range(4)]
        self.resetCoordinate = np.array([[146.5, 230.5, 360, 0], [60, 347, 275, 15], [375, 230.5, 360, 17]])
        self.distAvoid = np.array([np.inf, np.inf])
        # Receive from socket
        self.cameraCmd = {'steer': 0, 'speed': 230, 'parkingMode': 'False', 'avoidMode': 'False', "stopLine": "False", "stopSign": "False"}
        self.trafficCmd = {}
        self.coordinate = {'X': 0, "Y": 0, "Z": 0}
        # Receive Keyboard command using socket
        self.autopilot = False
        self._reset = False
        self.manualControl = None

        # Create path finder
        self.offsetDistance = 40
        self.pathHandling = PathHandler.clothoidsGenFile("./plannedCoor.txt"); self.pathHandling.generateOffsetPath(self.offsetDistance)
    
    def setupConnection(self):
        def get_wlan0_ip():
            addrs = psutil.net_if_addrs()
            wlan = addrs.get('wlan0')
            if wlan:
                for addr in wlan:
                    if addr.family == socket.AF_INET:
                        return addr.address
            return None
        # Read message from socket
        localhost = "127.0.0.1"
        assignedAddr = get_wlan0_ip()
        parkingPort = 8002
        laneGuidePort = 8001
        trafficLightPort = 5000
        coordinatePort = 5001
        keyboardPort = 8003
        

        # local socket
        realsenseSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        realsenseSocket.bind((localhost, laneGuidePort))
        picamSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        picamSocket.bind((localhost, parkingPort))
        keyboardSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        keyboardSocket.bind((localhost, keyboardPort))

        # Socket from outside
        trafficLightSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        trafficLightSocket.bind((assignedAddr, trafficLightPort))
        coordinateSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        coordinateSocket.bind((assignedAddr, coordinatePort))

        keyboardThread = threading.Thread(target = self.readKey, args = (keyboardSocket, ), daemon = True)
        keyboardThread.start()
        
        sockets = [[realsenseSocket, self.postprocessCamera], 
                   [picamSocket, self.postprocessCamera], 
                   [trafficLightSocket, self.postprocessTraffic],
                   [coordinateSocket, self.postprocessCoordinate]]
        threads = [threading.Thread(target = self.readMessage, args = (sock, funct, ), daemon = True) for sock, funct in sockets]        
        for t in threads:
            t.start()

        
    # =============================== STOP ================================================
    def stop(self):
        """Stop the thread and clean up resources."""
        self.stopEvent.set()  # Signal the thread to stop
        super(threadNucleoCmd, self).stop()
    

    # ================================ RUN ================================================
    def run(self):
        t = self.pathHandling.cameraSwitch

        # INIT CONTROLLER
        referenceVelocity = 19
        l = 26
        Ld = 65; self.Ld = Ld
        self.controller = PDPPController(Ld, l)
        self.lheFinder = LHEFinder(Ld, self.pathHandling.definedPath)
        self.lheFinderOffsetPath = LHEFinder(Ld, self.pathHandling.offsetPath)
        self.commandCenter = CommandSystem(25, Ld)
        
        # INIT LOGIC FOR RESET POSITION 
        foundStopSign = False; stopSignCnt = 0; prevStopSignCnt = 0

        # INITIALIZE POSITION 
        self.resetDynamic(self.resetCoordinate[0])
        prepRun = time.time()

        self.inAvoidance = False; self.exitCreated = False; self.relyCamera = False

        
        while self._running:
            if self._reset:
                if self.debugger: self.logger.info(f"  RESETING")
                self.lheFinder.currLookahead = 0; self.lheFinder.currPositionIdx = 0
                self.lheFinderOffsetPath.currLookahead = 0; self.lheFinderOffsetPath.currPositionIdx = 0
                self.resetDynamic(self.resetCoordinate[0])
                self.commandCenter.stopFlag = False
                self.commandCenter.switchFlag = False
                self.commandCenter.nextMajorPoint = 1
                self.commandCenter.prepareStopPoint = None
                self.commandCenter.switchPoint = None
                self.commandCenter.prepareStopFlag = False
                self._reset = False
                self.inAvoidance = False
                self.exitCreated = False
                
            if self.autopilot == False:
                self.manualFormat
                prepRun = time.time()
            else:
                # self.vehicleX = self.states[0] + self.initDynamic[0]
                # self.vehicleY = self.states[1] + self.initDynamic[1]
                self.vehicleX = self.coordinate['X']
                self.vehicleY = self.coordinate['Y']
                self.psi = self.states[2] * np.pi / 180
                
                position = np.array([self.vehicleX, self.vehicleY])[..., None]
                
                distance = self.pathHandling.distance2Path(position)
                distance2MajorPoint = self.pathHandling.distance2MajorPoint(position)
                self.lheFinder.updateState(distance)
                self.lheFinderOffsetPath.updateState(np.linalg.norm(np.array([self.vehicleX, self.vehicleY])[..., None] - self.pathHandling.offsetPath[:2], axis = 0))

                self.commandCenter.switchControl(self.pathHandling.definedPath[:, np.argmin(distance)], distance2MajorPoint)

                
                # Got to the Dynamic point
                if self.commandCenter.nextMajorPoint < len(t) and t[self.commandCenter.nextMajorPoint - 1]:
                    self.relyCamera = False
                # Got to the Camera point
                elif self.commandCenter.nextMajorPoint < len(t) and t[self.commandCenter.nextMajorPoint - 1] == False:
                    self.relyCamera = True
                
                
                steerGlobal, _ = self.globalControl(referenceVelocity, self.pathHandling.definedPath, 
                                                    distance, distance2MajorPoint)
                steerLocal, self.speedParam = self.camControl()
                finalSteer = fuser.update(np.deg2rad(steerGlobal), np.deg2rad(steerLocal), 
                                          self.relyCamera and not self.inAvoidance and bool(int(self.cameraCmd['have_lane'])), 
                                          float(self.cameraCmd['trust_left']), 
                                          float(self.cameraCmd['trust_right']), 
                                          dt = 0.05)[0]
                self.steerParam = int((np.rad2deg(finalSteer) / 30) * 230)
                
                if self.debugger: 
                    self.logger.debug(f"  Local: {steerLocal:.2f}, Global: {steerGlobal:.2f}")
                    self.logger.debug(f"  FINAL STEERING: {np.rad2deg(finalSteer):.2f}")
                    self.logger.debug(f"  Rely on camera: {self.relyCamera} | Have lanes: {bool(int(self.cameraCmd['have_lane']))}")
                
                    

                # =============== Check for intersection ==================
                if self.commandCenter.nextMajorPoint < len(t) and (t[self.commandCenter.nextMajorPoint - 1] == False and t[self.commandCenter.nextMajorPoint] == True) and distance2MajorPoint[self.commandCenter.nextMajorPoint] < 40: 
                    speed = self.intersectionCheck()
                    self.speedParam = self.speedParam if speed is None else speed
                
                # =============== Check Reset position ====================
                if self.cameraCmd['stopSign'] == 'True' and foundStopSign == False:
                    stopSignCnt += 1
                    foundStopSign = True 
                elif self.cameraCmd['stopSign'] == 'False': foundStopSign = False

                if prevStopSignCnt != stopSignCnt:
                    dist2ResetPoint = np.linalg.norm(np.array([self.vehicleX, self.vehicleY])[None, ...] - self.resetCoordinate[:, :2], axis = 1)
                    candidateResetIdx = np.argmin(dist2ResetPoint)
                    self.resetDynamic(self.resetCoordinate[candidateResetIdx])
                    prevStopSignCnt = stopSignCnt
                

                if self.cameraCmd['parkingMode'] == 'True':
                    self.sendCommand(self.commands['5'].format(0))
                    time.sleep(2)
                    self.parking()
                    time.sleep(2)
                    self.unpacking()
                    
                if self.debugger: self.logger.info(f"  States: {[f'{v:.2f}' for v in [self.vehicleX, self.vehicleY, self.psi * 180 / np.pi]]}")
                
                
                # Stop the car at the end of path
                if self.commandCenter.stopFlag == True:
                    self.speedParam = 0
                print(self.speedParam)
                    
                self.sendCommand(self.commands['6'].format(self.steerParam))
                if time.time() - prepRun > 5:
                    time.sleep(.02)
                    self.sendCommand(self.commands['5'].format(int(self.acceleration())))
                time.sleep(.02)
            
    # =============================== START ===============================================
    def start(self):
        super(threadNucleoCmd, self).start()

    # ============================ CONN READER ============================================
    def readKey(self, socketer: socket.socket):
        while True:
            
            data, _ = socketer.recvfrom(1024)
            message = data.decode("utf-8")
            splitCmd = message.split("|")

            if "Autopilot" in splitCmd:
                self.autopilot = True
            elif "Manual" in splitCmd:
                self.autopilot = False
            elif splitCmd[0] in ["Forward", "Left", "Right", "Backward", "Stop"]:
                self.manualControl = splitCmd[0]
            elif "Reset" in splitCmd:
                self._reset = True
                
    def readMessage(self, socketer: socket.socket, afterProc: None):
        while True:
            data, addr = socketer.recvfrom(1024)    
            message = data.decode("utf-8")
            splitCmd = message.split("|")
            
            afterProc(splitCmd)
    
    def postprocessCoordinate(self, splitCmd: list):
        for unprocessedCmd in splitCmd:
            key, value = unprocessedCmd.split(": ")
            self.coordinate[key] = float(value) * 100
    
    def postprocessCamera(self, splitCmd: list):
        for unprocessedCmd in splitCmd:
            key, value = unprocessedCmd.split(": ")
            self.cameraCmd[key] = value
        
        # if self.debugger: self.logger.debug(self.cameraCmd)
    
    def postprocessTraffic(self, splitCmd: list):
        id = None; foundId = False
        for unprocessedCmd in splitCmd:
            key, value = unprocessedCmd.split(": ")
            if key == "id":
                id = value
                foundId = True
                self.trafficCmd.update({int(id): {}})
            
            if foundId and key != 'id':
                self.trafficCmd[int(id)].update({key: value})
        
    def sendCommand(self, command: str):
        self.ser.write(command.encode())

    def readResponse(self):
        while True:
            time.sleep(.01)
            if self.ser.in_waiting > 0:
                response = self.ser.readline().decode('utf-8').strip()
                try:
                    if "x: " in response:
                        for idx, coorString in enumerate(response.split(", ")):
                            self.states[idx] = float(coorString.split(": ")[-1]) / 10
                        self.states[2] *= 10
                            
                    # if "x_uwb: " in response:
                    #     for idx, coorString in enumerate(response.split(", ")):
                    #         self.statesUWB[idx] = float(coorString.split(": ")[-1])
                    #     if time.time() - prevUWBTime <= 3:
                    #         self.initUWB.append(self.statesUWB[:])
                    #     else: collectedUWB = True
                    
                    if "distance:" in response:
                        distances = response.split(":")[-1]
                        for idx, dist in enumerate(distances.split(" ")):
                            self.distAvoid[idx] = int(dist)
                except: ...
    # ========================= MANUAL FORMAT ==========================================
    @property
    def manualFormat(self):
        if self.manualControl is not None:
            if self.manualControl == "Forward":
                self.speedParam += 20
                if self.speedParam > 490:
                    self.speedParam = 490
                self.acceleration(wait = True)

            if self.manualControl == "Backward":
                self.speedParam -= 20
                if self.speedParam < -490:
                    self.speedParam = -490
                self.acceleration(wait = True)

            if self.manualControl == "Stop":
                self.speedParam = 0
                self.acceleration(wait = True)
            
            if self.manualControl == "Right":
                self.steerParam += 25
                if(self.steerParam > 230):
                    self.steerParam = 230  
                self.sendCommand(self.commands['6'].format(self.steerParam))

            if self.manualControl == "Left":
                
                self.steerParam -= 25
                if(self.steerParam < -230):
                    self.steerParam = -230  
                self.sendCommand(self.commands['6'].format(self.steerParam))
            
            time.sleep(0.1)
            
            if self.debugger: self.logger.info(f"  Manual Control: {self.manualControl} - {self.steerParam if self.manualControl in ['Right', 'Left'] else self.speedParam}")    
            self.manualControl = None

    #========================== HELPER FUNCT ===========================================
    def resetDynamic(self, resetCoor: list):
        time.sleep(1)
        prevDynamicTime = time.time()
        self.sendCommand(self.commands['5'].format(0))
        for _ in range(3):
            angle = int(resetCoor[2])
            self.sendCommand(self.commands['8'].format(angle))
            time.sleep(0.5)

        while True:
            if time.time() - prevDynamicTime >= 1:
                self.initDynamic[0] = - self.states[0] + resetCoor[0]
                self.initDynamic[1] = - self.states[1] + resetCoor[1]
                self.initDynamic[2] = self.states[0] * 10
                break
    
    
    
    # =============================== CONTROL VIA CAM ============================
    def camControl(self) -> Tuple[float, float]:
        velocity = float(self.cameraCmd['speed'])
        speedParam = velocity * 10

        angle = float(self.cameraCmd['steer'])

        return angle, speedParam
        
    # ============================== CONTROL VIA GLOBAL ==========================
    def globalControl(self, refVelocity: float, definedPath: np.ndarray, 
                      distance: np.ndarray, dist2Major: np.ndarray) -> Tuple[float, float]:
        alpha = self.lheFinder.LHE(self.vehicleX, self.vehicleY, self.psi)

        forwardObjDetection = self.cameraCmd['avoidMode']
        starboardObjDetection = np.any(self.distAvoid < 350)

        if self.debugger: self.logger.debug(f"{forwardObjDetection}, {starboardObjDetection}, {self.distAvoid }")

        if forwardObjDetection == "True" and not self.inAvoidance:
            self.inAvoidance = True
            self.lastIdxEntryRamp = self.pathHandling.entryRamp(self.lheFinder.currPositionIdx, 
                                                                self.lheFinderOffsetPath.currPositionIdx, 
                                                                possibleEnd = 40, 
                                                                offsetOnPath=5, 
                                                                offsetDist=40)
            self.lheFinderOffsetPath.updatePath(self.pathHandling.offsetPath)
            
        
        if self.inAvoidance:
            self.cameraCmd['have_lane'] == "False"
            if (self.lheFinderOffsetPath.currPositionIdx < self.lastIdxEntryRamp or starboardObjDetection): 
                alpha = self.lheFinderOffsetPath.LHE(self.vehicleX, self.vehicleY, self.psi)
                if self.debugger: self.logger.debug("  IN OFFSET")
            if self.lheFinderOffsetPath.currPositionIdx > self.lastIdxEntryRamp and starboardObjDetection == False and self.exitCreated == False:
                if self.debugger: self.logger.debug("  EXITING")
                self.lastIdxExitRamp = self.pathHandling.exitRamp(self.lheFinder.currPositionIdx, offsetOnPath=5, offsetDist=40)
                self.lheFinder.updatePath(self.pathHandling.definedPath)
                self.exitCreated = True
            
            if self.exitCreated == True and self.lheFinder.currPositionIdx > self.lastIdxExitRamp:
                self.inAvoidance = False
                self.exitCreated = False
        
        
        angle = self.controller.PDcontrol(alpha, 0.9, 0)
        angle = np.clip(angle * 180 / np.pi, -30, 30)
        steerParam = -(int(angle) / 30) * 230            
        velocity = self.commandCenter.stopControl(refVelocity, definedPath[:, np.argmin(distance)], dist2Major)
        speedParam = velocity * 10
        
        return -angle, speedParam
    
    # ============================= INTERSECTION CHECKING ========================
    def intersectionCheck(self) -> float:
        for id, state in self.trafficCmd.items():
            trafficLightCoor = np.array([float(state['x']), float(state['y'])])
            trafficLightState = int(state['state'])
        
            dist2StopLight = np.linalg.norm(trafficLightCoor - np.array([self.vehicleX, self.vehicleY]))
            if dist2StopLight < 60:
                if (trafficLightState == 0 or trafficLightState == 1) and self.cameraCmd['stopLine'] == 'True':
                    velocity = 0
                else: velocity = 12
                return velocity * 10

    # ========================== PARKING AND AVOID ===============================    

    def unpacking(self):
        time.sleep(2)
        self.sendCommand(self.commands['5'].format(-90))
        time.sleep(3)
        self.sendCommand(self.commands['5'].format(0))
        time.sleep(2)
        self.sendCommand(self.commands['6'].format(-225))
        time.sleep(2)
        self.sendCommand(self.commands['5'].format(100))
        time.sleep(8)
        self.sendCommand(self.commands['5'].format(0))
        self.sendCommand(self.commands['6'].format(225))
        time.sleep(7.5)
        self.sendCommand(self.commands['6'].format(0))
        time.sleep(0.1)
        self.sendCommand(self.commands['5'].format(0))

    def parking(self):
        self.sendCommand(self.commands['6'].format(230))
        time.sleep(2)
        self.sendCommand(self.commands['5'].format(-90))
        time.sleep(7.8)
        self.sendCommand(self.commands['6'].format(-230))
        time.sleep(2)
        self.sendCommand(self.commands['5'].format(-90))
        time.sleep(5.3)
        self.sendCommand(self.commands['6'].format(50))
        time.sleep(0.1)
        self.sendCommand(self.commands['5'].format(100))
        time.sleep(3.5)
        self.sendCommand(self.commands['5'].format(0))
        time.sleep(0.1)
        self.sendCommand(self.commands['6'].format(0))
    
    
    def acceleration(self, wait = False) -> None:
        if wait == False:
            increment = 7
            diff = self.speedParam - self.actualSpeed

            if abs(diff) > increment:
                self.actualSpeed += increment if diff > 0 else -increment
            else:
                self.actualSpeed = self.speedParam

            return self.actualSpeed
        else:
            increment = 7
            
            while True:
                diff = self.speedParam - self.actualSpeed

                if abs(diff) > increment:
                    self.actualSpeed += increment if diff > 0 else -increment
                    self.sendCommand(self.commands['5'].format(int(self.actualSpeed)))
                else:
                    self.actualSpeed = self.speedParam
                    self.sendCommand(self.commands['5'].format(int(self.actualSpeed)))
                    break
                
                time.sleep(0.02)
                