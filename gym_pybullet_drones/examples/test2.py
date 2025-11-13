import torch
import time
import pybullet as p
import numpy as np

# PPO 모델과 커스텀 환경을 불러옵니다.
from stable_baselines3 import PPO
from moving_car_test import make_custom_env
from gym_pybullet_drones.utils.utils import sync
# (VecNormalize import 제거)

# 1. 불러올 모델 파일의 경로
MODEL_PATH = "./models_multi/best_model.zip"    #학습 도중 가장 성능이 좋았던 모델 파일 경로
# (VEC_NORMALIZE_PATH 제거)

# 2. 학습할 때 사용했던 observation mode와 동일하게 설정해야 합니다.
OBS_MODE = "rel_pos"


if __name__ == "__main__":
    # --- 환경 생성 ---
    # (VecEnv 래퍼 제거)
    env = make_custom_env(gui=True, obs_mode=OBS_MODE, is_test_mode=True)
    
    # (VecNormalize.load() 관련 코드 4줄 제거)

    # (롤백) unwrapped를 통해 원본 환경 속성에 직접 접근
    TIMESTEP = env.unwrapped.CTRL_TIMESTEP  # 시뮬레이션 타임스텝 (초)
    
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
    # (롤백) .reset()은 2개의 값을 반환합니다.
    obs, info = env.reset()
    print("테스트를 시작합니다...")
    
    # 10 에피소드 동안 테스트
    for episode in range(10):
        # --- FSM(상태 기계) 변수 초기화 ---
        STATE_TRACKING = 0
        STATE_SEARCHING = 1
        current_state = STATE_TRACKING
        search_timer = 0
        
        # --- FSM 하이퍼파라미터 ---
        max_search_time = 240 # 최대 탐색 시간 (10초 @ 24Hz)
        REACQ_ANGLE_THRESHOLD = 45.0 # 재획득 카메라 시야각 (도)

        # --- 에피소드 변수 초기화 ---
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        start = time.time()
        # (롤백) .reset()은 2개의 값을 반환합니다.
        obs, info = env.reset()

        # (롤백) unwrapped를 통해 원본 환경 속성에 직접 접근
        drone_id = env.unwrapped.DRONE_IDS[0]
        client_id = env.unwrapped.CLIENT # (카메라 추적용 ID 미리 저장)

        while not (terminated or truncated):
            
            # 1. FSM 상태에 따라 행동(action) 결정
            if current_state == STATE_TRACKING:
                # (DRL 모드) PPO 모델의 결정 사용
                action, _states = model.predict(obs, deterministic=True)
            
            elif current_state == STATE_SEARCHING:
                # (탐색 모드) "타겟 방향으로 상승 비행" 액션 생성
                search_timer += 1
                
                normalized_direction = np.array([0.0, 0.0, 1.0]) # (Failsafe: 수직 상승)
                    
                throttle = 0.1 #  속도로 탐색 비행
                
                action = np.array([
                    0.0, # target_vx (방향)
                    0.0, # target_vy (방향)
                    1.0, # target_vz (방향)
                    throttle               # 속도 크기
                ])

            # 2. 결정된 행동으로 환경 실행
            # (롤백) .step()은 5개의 값을 반환합니다.
            action = np.array(action, dtype=np.float32)
            if action.ndim == 1:
                action = action.reshape(1, -1)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # (롤백) reward는 스칼라 값입니다.
            total_reward += reward
            steps += 1

            # 3. FSM 상태 전이 로직
            # (롤백) info는 딕셔너리입니다.
            status = info.get('status', None) # moving_car_test.py가 보낸 신호

            if current_state == STATE_TRACKING:
                if status == 'TARGET_LOST':
                    #1. 추적 -> 탐색 모드로 바뀔 때 출력
                    print(f"[FSM] Target lost (Step {steps}). Switching to SEARCH mode.")
                    current_state = STATE_SEARCHING
                    search_timer = 0
                    
            elif current_state == STATE_SEARCHING:
                
                # --- 각도 기반 재획득 조건 검사 ---
                drone_down_vector = np.array([0.0, 0.0, -1.0])
                # (롤백) obs는 배열입니다.
                drone_to_target_vector = obs[0:3] # rel_pos
                dist_3d = np.linalg.norm(drone_to_target_vector)
                
                reacq_angle_deg = 90.0 # (기본값)
                
                if dist_3d > 1e-6:
                    dot_product = -drone_to_target_vector[2] # (0,0,-1) . (x,y,z) = -z
                    cos_theta = np.clip(dot_product / dist_3d, -1.0, 1.0)
                    reacq_angle_deg = np.degrees(np.arccos(cos_theta))

                if reacq_angle_deg < REACQ_ANGLE_THRESHOLD:
                    #2. 탐색 -> 추적 모드로 바뀔 때 출력
                    print(f"[FSM] Target Re-acquired! (Angle: {reacq_angle_deg:.2f}°). Switching to TRACKING mode.")
                    current_state = STATE_TRACKING
                    
                elif search_timer > max_search_time:
                    #3. 탐색 실패(타임아웃) 시 출력
                    print(f"[FSM] Search Failed (timeout). Terminating episode.")
                    terminated = True # 에피소드 강제 종료
            
            # --- 카메라 추적 로직 ---
            # (롤백) 미리 저장한 client_id 사용
            drone_pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=client_id)
            
            p.resetDebugVisualizerCamera(
                cameraDistance=3,      
                cameraYaw=0,           
                cameraPitch=-70,        
                cameraTargetPosition=drone_pos 
            )

            sync(steps, start, TIMESTEP)

        print(f"에피소드 {episode + 1}: 총 보상 = {total_reward:.2f}, 스텝 수 = {steps}")
        # (롤백) .reset()은 2개의 값을 반환합니다.
        obs, info = env.reset()

    env.close()
    print("테스트가 종료되었습니다.")