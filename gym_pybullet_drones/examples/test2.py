import torch
import time
import pybullet as p

# PPO 모델과 커스텀 환경을 불러옵니다.
from stable_baselines3 import PPO
from moving_car_test import make_custom_env
from gym_pybullet_drones.utils.utils import sync

# 1. 불러올 모델 파일의 경로
MODEL_PATH = "ppo_drone_multi_final.zip"

# 2. 학습할 때 사용했던 observation mode와 동일하게 설정해야 합니다.
OBS_MODE = "rel_pos"


if __name__ == "__main__":
    # --- 환경 생성 ---
    # 테스트 시에는 시뮬레이션 창을 봐야 하므로 gui=True로 설정합니다.
    env = make_custom_env(gui=True, obs_mode=OBS_MODE)

    TIMESTEP = env.CTRL_TIMESTEP  # 시뮬레이션 타임스텝 (초)
    # 디버그용 카메라의 위치와 각도를 설정합니다.
    p.resetDebugVisualizerCamera(
        cameraDistance=5,      # 카메라와 타겟 사이의 거리 (값을 키울수록 멀어짐)
        cameraYaw=45,           # 카메라의 수평 회전 각도 (정면 = 0)
        cameraPitch=-30,        # 카메라의 수직 기울기 (위에서 아래로 보는 각도)
        cameraTargetPosition=[0, 0, 0] # 카메라가 바라보는 지점 (월드 좌표)
    )
    # --- 모델 불러오기 ---
    # device='auto'로 설정하면 사용 가능한 GPU를 자동으로 찾습니다.
    model = PPO.load(MODEL_PATH, env=env, device='auto')
    print(f"'{MODEL_PATH}' 모델을 성공적으로 불러왔습니다.")

    # --- 테스트 루프 실행 ---
    obs, info = env.reset()
    print("테스트를 시작합니다...")
    
    # 10 에피소드 동안 테스트
    for episode in range(10):
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        start = time.time()
        obs, info = env.reset()

        while not (terminated or truncated):
            # deterministic=True: 학습된 정책이 가장 좋다고 생각하는 행동을 항상 선택합니다.
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            sync(steps, start, TIMESTEP)

        print(f"에피소드 {episode + 1}: 총 보상 = {total_reward:.2f}, 스텝 수 = {steps}")
        obs, info = env.reset()

    env.close()
    print("테스트가 종료되었습니다.")