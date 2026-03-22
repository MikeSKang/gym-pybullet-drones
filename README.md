
# Tracking Drone Project 🚁

이 프로젝트는 [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones) 환경을 기반으로 강화학습(PPO)을 사용하여 드론이 목표물을 추적(Tracking)하도록 훈련하고 테스트하는 프로젝트입니다.

## 📌 주요 기능

프로젝트는 크게 두 가지 테스트 모드를 제공합니다:

1.  **State-based Tracking (`test2.py`)**: 드론과 타겟의 Relative Position 정보를 직접 받아 추적합니다. FSM(Finite State Machine)을 통해 놓쳤을 때 수색(Searching) 모드로 전환하는 로직이 포함되어 있습니다.
2.  **Vision-based Tracking (`test3.py`)**: 드론의 RGB 카메라 이미지를 입력으로 사용하며, **Siamese Network**를 통해 타겟의 위치를 추정하고 추적합니다.

## 🛠️ Installation

### 1\. 환경 설정

Python 3.10 환경을 권장합니다.

```bash
conda create -n drone_tracking python=3.10
conda activate drone_tracking
```

### 2\. Requirements

이 프로젝트는 `gym-pybullet-drones` 라이브러리에 의존합니다.

**수동 설치**
주요 의존성 패키지는 다음과 같습니다.

```bash
pip install gymnasium==0.29.1 pybullet==3.2.7 stable-baselines3==2.7.0 torch==2.8.0 opencv-python==4.10.0 numpy matplotlib
```

추가적으로 `gym-pybullet-drones` 원본 리포지토리를 clone 후 설치해야 할 수 있습니다.

## 📂 Directory Structure

실행을 위해 아래와 같은 폴더 구조와 모델 파일이 준비되어 있어야 합니다.

```
project_root/
├── requirements.txt
├── 2400best/                  # test2.py용 모델 폴더
│   ├── best_model.zip
│   └── final_vecnormalize.pkl (또는 vecnorm 파일들)
├── results_learn3/            # test3.py용 모델 폴더
│   ├── best_model.zip
│   └── final_vecnormalize.pkl
├── gym_pybullet_drones/       # 라이브러리 및 예제 코드
│   └── examples/
│       ├── test2.py           # State 기반 추적 실행 파일
│       ├── test3.py           # Vision 기반 추적 실행 파일
│       ├── BaselinePretrained.pth.tar  # Siamese Network 사전 학습 모델
│       └── tracking_object.png         # 추적 템플릿 이미지
└── ...
```

## 🚀 Usage

### 1\. State-based Tracking 실행

상대 좌표를 기반으로 학습된 PPO 모델을 테스트합니다. FSM 로직이 포함되어 있어 타겟을 놓치면 제자리에서 회전하며 수색합니다.

```bash
# gym_pybullet_drones/examples 폴더 내에서 실행
python test2.py
```

  * **모델 경로:** `./2400best/best_model.zip`
  * **특징:** `moving_car_test` 환경 사용, `OBS_MODE="rel_pos"`

### 2\. Vision-based Tracking (Siamese Network) 실행

RGB 이미지를 입력으로 받아 Siamese Network가 타겟을 인식하고, PPO 에이전트가 이를 추적합니다. 실행 시 OpenCV 윈도우를 통해 드론의 시야와 인식된 Heatmap을 확인할 수 있습니다.

```bash
# gym_pybullet_drones/examples 폴더 내에서 실행
python test3.py
```

  * **모델 경로:** `./results_learn3/best_model.zip`
  * **Siamese 모델:** `BaselinePretrained.pth.tar`
  * **특징:** OpenCV HUD 시각화 (Conf 점수, 속도 벡터 표시), `OBS_MODE="rgb"`

## 📎 References

  * **Base Environment:** [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
  * **RL Algorithm:** [Stable-Baselines3 PPO](https://github.com/DLR-RM/stable-baselines3)
