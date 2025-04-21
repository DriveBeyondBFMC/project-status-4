# Task 1: Lane changing
## 📂 Project Directory Structure
```bash
~/project-status-4/src/Task_1
├── PathGenerator.py
└── threadNucleoCmd.py
```

## 📝 Description
   Focuses on the lane changing functionality for moving vehicle scenarios

**Path and offset path generator**  
   - File: `PathGenerator.py`
   - Functionality: Parses the input nodes to build the primary trajectory while simultaneously generating an offset path for vehicle avoidance.

**Nucleo command Module**
   - File: `threadNucleoCmd.py`
   - Functionality: Receives steering from camera, calculate steering from path following and fusing both input to a unified steering angle. Sending both velocity and steering to STM Nucleo


# Task 2: New controller evaluation
## 📂 Project Directory Structure
```bash
~/project-status-4/src/Task_2
└── MPCControl.py
```
## 📝 Description
   Evaluate stability of MPC controller for path following task

**MPC Controller**
   - File: `MPCControl.py`
   - Functionality: Evaluate the performance of MPC controller on path following task.