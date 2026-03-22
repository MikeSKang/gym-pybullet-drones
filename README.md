# Vision-Based Dynamic Target Tracking System for UAVs 🚁

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-EE4C2C)
![Stable-Baselines3](https://img.shields.io/badge/Stable_Baselines3-PPO-20B2AA)
![PyBullet](https://img.shields.io/badge/Physics-PyBullet-lightgrey)

## 📌 Overview
This repository entails the implementation of an **Autonomous UAV Tracking System** designed to track a dynamically moving Unmanned Ground Vehicle (UGV) exclusively utilizing onboard vision data within a physics-based simulation. To surmount the limitations of conventional discrete control, we adopted a continuous control framework utilizing **Proximal Policy Optimization (PPO)**. Furthermore, we explored the feasibility of autonomous navigation in GPS-denied environments through a modular architecture incorporating **Siamese Networks**.

## 🚀 Research Methodology & Two-Phase Approach

This project was systematically conducted in two discrete phases, progressing from empirical validation to realistic vision-based implementation.

### Phase 1: Coordinate-Based Tracking Validation (`learn2.py`, `test2.py`)
This phase focuses on verifying whether the reinforcement learning algorithm can acquire a stable control policy in a highly realistic physics engine.
* **Approach:** The agent is trained using the absolute coordinates (Ground Truth) of the target.
* **Objective:** To secure smooth flight trajectories via a continuous action space and verify the convergence of the algorithm.
* **Result:** Achieved a tracking success rate of approximately 96.5%, substantiating the efficacy of the PPO algorithm in a physics-based environment.

### Phase 2: Vision-Based Modular Tracking (`learn3.py`, `test3.py`)
In this final phase, absolute coordinate inputs were eliminated to enable target tracking relying solely on the UAV's onboard RGB camera, simulating real-world constraints.
* **Perception:** A Siamese Network analyzes the RGB images to compute the relative coordinates of the target in real-time.
* **Control:** Based on the estimated coordinates, the PPO agent deduces and outputs optimal velocity commands.
* **Hybrid Strategy:** To mitigate 'Target Lost' scenarios where the target escapes the camera's Field of View (FOV), we integrated a heuristic, rule-based searching mechanism, thereby enhancing the overall robustness of the system.

## 🧠 System Architecture

### 1. Control Module (Reinforcement Learning)
* **Algorithm:** PPO via Stable-Baselines3.
* **State Space (16D Vector):** Incorporates relative position, relative velocity, the UAV's angular velocity and attitude (RPY), alongside the previous action vector.
* **Action Space (4D Continuous):** Velocity commands spanning $(v_x, v_y, v_z, \omega_z)$.
* **Reward Function:** Comprises a distance maintenance reward (Exponential Kernel) and imposes stringent penalties for angular instability, abrupt control shifts, and ground proximity.

### 2. Vision Module (Siamese Networks)
* **Model:** SiamFC-based architecture.
* **Features:** Utilizes a pre-trained model for zero-shot target discrimination, demonstrating robust detection capabilities even in environments with unseen textures and visual distractors.

## 📂 Project Structure
```text
📦 gym-pybullet-drones
 ┣ 📂 2400best                 # Core training and evaluation scripts
 ┃ ┣ 📜 learn2.py / test2.py   # [Phase 1] Ground Truth-based training/eval
 ┃ ┗ 📜 learn3.py / test3.py   # [Phase 2] Vision-based training/eval
 ┣ 📂 gym_pybullet_drones      # Core physics engine and custom environment wrappers
 ┃ ┗ 📂 envs/HoverAviary.py    # Base UAV environment definitions
 ┗ 📜 README.md
```

## 📊 Results & Discussion (Lessons Learned)
During Phase 2, the integration of the Vision Module yielded a tracking success rate ranging from **68.5% to 83.5% (with Hybrid Control)**. This marginal decline from Phase 1 elucidated several crucial academic insights:

* **Information Accuracy vs. Realism:** We observed a distinct trade-off between the precision of Ground Truth data and the uncertainties inherent in vision-based estimation. Bridging this gap is an indispensable step for real-world (Sim-to-Real) deployment.
* **Impact of Vision Noise:** The minute fluctuations (vision noise) in the coordinates estimated by the Siamese Network significantly impeded the control policy. This underscores the necessity of signal stabilization techniques, such as a Kalman Filter, for future iterations.
* **Efficacy of Hybrid Control:** Supplementing the pure reinforcement learning agent with a classical search algorithm effectively resolved the 'Target Lost' predicament, proving that combining RL with heuristic approaches substantially bolsters system resilience.

## 👥 Contributors
* **Kang Seung-hyun:** Control Module design (PPO), reward function formulation, and Hybrid Control Strategy implementation.
* **Hwang Chan-hong:** Vision Module design (Siamese Network) and perception-control pipeline integration.
