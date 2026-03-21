'''
Test2: 절대좌표에서 ppo만으로 학습된 모델로 이동하는 타겟 추적 테스트
FSM 기반 타겟 추적 테스트
'''
import cv2
import os
import time
import numpy as np
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from moving_car_test import make_custom_env
from gym_pybullet_drones.utils.utils import sync

def draw_4way_hud(img, rel_vel_world, drone_yaw, conf_score=None):
    """
    img: 카메라 이미지
    rel_vel_world: 타겟의 상대 속도(월드 좌표)
    drone_yaw: 드론의 현재 헤딩 각도
    conf_score: 샴 네트워크의 확신도 점수 (float)
    """
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # ---------------------------------------------------------
    # 1. 좌표 변환
    # ---------------------------------------------------------
    vx_world, vy_world = rel_vel_world[0], rel_vel_world[1]
    vel_forward = vx_world * np.cos(drone_yaw) + vy_world * np.sin(drone_yaw)
    vel_right = -vx_world * np.sin(drone_yaw) + vy_world * np.cos(drone_yaw)

    # ---------------------------------------------------------
    # 2. 4방향 바 그리기
    # ---------------------------------------------------------
    scale = 30.0; max_len = 100; bar_width = 10; gap = 20
    
    # 십자선
    cv2.line(img, (center_x, center_y - max_len - gap), (center_x, center_y + max_len + gap), (100, 100, 100), 1)
    cv2.line(img, (center_x - max_len - gap, center_y), (center_x + max_len + gap, center_y), (100, 100, 100), 1)
    
    # 상하좌우 바
    if vel_forward > 0: # 전진 (Red)
        length = int(min(abs(vel_forward) * scale, max_len))
        cv2.rectangle(img, (center_x - bar_width//2, center_y - gap - length), (center_x + bar_width//2, center_y - gap), (0, 0, 255), -1)
    elif vel_forward < 0: # 후진 (Yellow)
        length = int(min(abs(vel_forward) * scale, max_len))
        cv2.rectangle(img, (center_x - bar_width//2, center_y + gap), (center_x + bar_width//2, center_y + gap + length), (0, 255, 255), -1)

    if vel_right > 0: # 우측 (Green)
        length = int(min(abs(vel_right) * scale, max_len))
        cv2.rectangle(img, (center_x + gap, center_y - bar_width//2), (center_x + gap + length, center_y + bar_width//2), (0, 255, 0), -1)
    elif vel_right < 0: # 좌측 (Green)
        length = int(min(abs(vel_right) * scale, max_len))
        cv2.rectangle(img, (center_x - gap - length, center_y - bar_width//2), (center_x - gap, center_y + bar_width//2), (0, 255, 0), -1)

    # ---------------------------------------------------------
    # 3. Conf 점수 및 상태 표시
    # ---------------------------------------------------------
    if conf_score is None:
        # Ground Truth 기반 테스트에서는 점수가 없으므로 회색으로 표시하거나 표시하지 않습니다.
        conf_text = "CONF: N/A (GT)"
        conf_color = (100, 100, 100) # 회색
    elif conf_score > 3.0:
        conf_text = f"CONF: {conf_score:.2f}"
        conf_color = (0, 255, 0) # Green (안전)
    elif conf_score > 2.0:
        conf_text = f"CONF: {conf_score:.2f}"
        conf_color = (0, 255, 255) # Yellow (주의)
    else:
        conf_text = f"CONF: {conf_score:.2f}"
        conf_color = (0, 0, 255) # Red (위험)

    # 좌측 상단에 Conf 표시
    # [수정] 위에서 결정된 텍스트와 색상을 사용합니다.
    cv2.putText(img, conf_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

    # 중앙 하단 상태 메시지
    font_scale = 0.5
    if abs(vel_forward) < 0.2 and abs(vel_right) < 0.2:
        msg = "LOCK-ON"
        msg_color = (0, 255, 0)
    else:
        msg = "TRACKING"
        msg_color = (0, 255, 255)
        
    cv2.putText(img, msg, (center_x - 30, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, msg_color, 2)
    
    return img

# =============================================================================
# 1. 경로 및 하이퍼파라미터 설정
# =============================================================================
MODEL_DIR = "./2400best"  # 모델이 저장된 폴더 경로
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
VECNORM_CANDIDATES = [
    os.path.join(MODEL_DIR, "vecnormalize.pkl"),
    os.path.join(MODEL_DIR, "latest_vecnormalize.pkl"),
    os.path.join(MODEL_DIR, "final_vecnormalize.pkl"),
]

OBS_MODE = "rel_pos"

# --- FSM(유한 상태 머신) 상수 ---
STATE_TRACKING = 0
STATE_SEARCHING = 1

# --- FSM 파라미터 ---
MAX_SEARCH_TIME_STEPS = 240   # 수색 최대 시간 (약 8초)
REACQ_ANGLE_THRESHOLD = 45.0  # 재획득 시야각 (도)

# =============================================================================
# 2. 환경 생성 함수
# =============================================================================
def make_env_for_test():
    # GUI 켜기, 테스트 모드 활성화
    return make_custom_env(gui=True, obs_mode=OBS_MODE, is_test_mode=True)

# =============================================================================
# 3. 메인 실행 블록
# =============================================================================
if __name__ == "__main__":
    # 1) DummyVecEnv로 환경 래핑
    vec_env = DummyVecEnv([make_env_for_test])

    # 2) VecNormalize 통계 로드 (학습된 정규화 수치 적용)
    vecnorm_loaded = False
    for path in VECNORM_CANDIDATES:
        if os.path.exists(path):
            print(f"[INFO] Loading VecNormalize stats from: {path}")
            vec_env = VecNormalize.load(path, vec_env)
            vecnorm_loaded = True
            break

    if not vecnorm_loaded:
        print("[WARN] VecNormalize .pkl not found! Running without normalization.")

    # 테스트 모드이므로 통계 업데이트 중지
    if isinstance(vec_env, VecNormalize):
        vec_env.training = False
        vec_env.norm_reward = False

    # 3) PPO 모델 로드
    model = PPO.load(MODEL_PATH, env=vec_env, device="auto")
    print(f"[INFO] Loaded model from {MODEL_PATH}")

    # 4) PyBullet 내부 객체 접근 (Ground Truth 계산용)
    # Wrapper를 뚫고 들어가 실제 ID들을 가져옵니다.
    wrapper = vec_env.venv.envs[0]
    
    # Gymnasium Wrapper 속성 접근 방식으로 ID 가져오기
    try:
        drone_id = wrapper.get_wrapper_attr('DRONE_IDS')[0]
        client_id = wrapper.get_wrapper_attr('CLIENT')
        # moving_car_test.py의 MovingTargetWrapper에 저장된 _target_id 접근
        target_id = wrapper.get_wrapper_attr('_target_id') 
    except AttributeError:
        # 구버전 호환성 (직접 접근)
        drone_id = wrapper.env.unwrapped.DRONE_IDS[0]
        client_id = wrapper.env.unwrapped.CLIENT
        target_id = getattr(wrapper.env.unwrapped, '_target_id', None)

    TIMESTEP = wrapper.get_wrapper_attr('CTRL_TIMESTEP')
    print(f"[INFO] Simulation Timestep: {TIMESTEP:.4f} sec")

    # 디버깅 카메라 초기화
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
        physicsClientId=client_id
    )

    print(">>> 테스트 시작 <<<")

    for ep in range(10):  # 10회 테스트
        done = False
        total_reward = 0.0
        steps = 0
        start_time = time.time()

        # FSM 초기화
        current_state = STATE_TRACKING
        search_timer = 0
        
        # 마지막으로 확인된 타겟의 '진짜' 상대 위치 (초기값: 전방 1m)
        last_known_true_rel_pos = np.array([1.0, 0.0, 0.0])

        obs = vec_env.reset()

        # 에피소드마다 타겟 ID가 바뀔 수 있으므로 갱신 시도
        try:
            target_id = wrapper.get_wrapper_attr('_target_id')
        except:
            pass

        while not done:
            # -----------------------------------------------------------
            # [Step 0] Ground Truth(진짜 좌표) 계산
            # 정규화된 obs 대신, 물리 엔진에서 직접 좌표를 가져와 판단합니다.
            # -----------------------------------------------------------
            d_pos, d_orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=client_id)
            t_pos, t_orn = p.getBasePositionAndOrientation(target_id, physicsClientId=client_id)
            print(d_pos)
            # 실제 상대 위치 벡터 (Target - Drone)
            true_rel_pos = np.array(t_pos) - np.array(d_pos)
            true_dist_3d = np.linalg.norm(true_rel_pos)

            # Tracking 중이라면 마지막 위치 기억 (수색용)
            if current_state == STATE_TRACKING:
                last_known_true_rel_pos = true_rel_pos

            # -----------------------------------------------------------
            # [Step 1] FSM 행동 결정 (Action Selection)
            # -----------------------------------------------------------
            if current_state == STATE_TRACKING:
                # 모델 예측 (obs는 정규화되어 있음 -> 모델에 적합)
                action, _ = model.predict(obs, deterministic=True)
            
            elif current_state == STATE_SEARCHING:
                search_timer += 1
                
                # 기억해둔 마지막 위치 방향 계산
                target_dx = last_known_true_rel_pos[0]
                target_dy = last_known_true_rel_pos[1]
                
                mag_xy = np.hypot(target_dx, target_dy)
                
                # 방향 벡터 정규화
                if mag_xy > 0.1: # 거리가 너무 가까우면 방향 의미 없음
                    dir_x = target_dx / mag_xy
                    dir_y = target_dy / mag_xy
                else:
                    dir_x, dir_y = 1.0, 0.0 # 정보 없으면 전방 수색
                
                # [수색 액션]
                # - XY: 방향 * 0.5 (적당한 속도로 이동)
                # - Z: 0.0 (고도 유지 -> 1.0에서 0.0으로 수정됨)
                # - Yaw Rate: 0.1 (천천히 회전하며 주변 살피기)
                # - Shape: (1, 1, 4)로 만들어 차원 에러 방지
                action = np.array([[[dir_x * 0.5, dir_y * 0.5, 0.0, 0.1]]], dtype=np.float32)

            # -----------------------------------------------------------
            # [Step 2] 환경 실행
            # -----------------------------------------------------------
            obs, reward, done, info = vec_env.step(action)

            # 데이터 언패킹
            reward_val = reward[0] if hasattr(reward, "__len__") else reward
            done_flag = done[0] if hasattr(done, "__len__") else done
            single_info = info[0] if isinstance(info, list) else info

            total_reward += reward_val
            steps += 1

            # -----------------------------------------------------------
            # [Step 3] FSM 상태 전이 (State Transition)
            # -----------------------------------------------------------
            # 환경에서 보내주는 상태 메시지 확인 (e.g. 시야각 벗어남)
            env_status = single_info.get('status', None)

            if current_state == STATE_TRACKING:
                if env_status == 'TARGET_LOST':
                    print(f"[FSM] 놓침! (Step {steps}). 마지막 위치: {last_known_true_rel_pos[:2].round(2)}. 수색 모드 전환.")
                    current_state = STATE_SEARCHING
                    search_timer = 0
            
            elif current_state == STATE_SEARCHING:
                # [핵심] Ground Truth 기반 재획득 판정
                # 정규화된 obs 대신 위에서 계산한 true_rel_pos 사용
                
                reacq_angle_deg = 90.0
                if true_dist_3d > 1e-6:
                    # 드론 기준 하방 벡터(0,0,-1)와 타겟 벡터 간 각도
                    # Dot( (0,0,-1), (x,y,z) ) = -z
                    dot_val = -true_rel_pos[2]
                    cos_theta = np.clip(dot_val / true_dist_3d, -1.0, 1.0)
                    reacq_angle_deg = np.degrees(np.arccos(cos_theta))
                
                # 판정
                if reacq_angle_deg < REACQ_ANGLE_THRESHOLD:
                    print(f"[FSM] 타겟 재발견! (각도: {reacq_angle_deg:.1f}°). Tracking 복귀.")
                    current_state = STATE_TRACKING
                
                elif search_timer > MAX_SEARCH_TIME_STEPS:
                    print(f"[FSM] 수색 시간 초과. 에피소드 종료.")
                    done_flag = True

            # 1. 환경에서 RGB 이미지 가져오기
            # (DummyVecEnv로 감싸져 있어서 unwrapped 접근 필요)
            try:
                # obs_mode가 'rgb' 또는 'both'여야 함
                rgb_image = wrapper.get_wrapper_attr('_render_rgb')() 
            except AttributeError:
                # 만약 _render_rgb 접근이 안 된다면, 직접 PyBullet API 호출 필요
                # (moving_car_test.py의 _render_rgb 함수 참조)
                print("[WARN] Cannot get RGB image. Skip HUD visualization.")
                rgb_image = None

            if rgb_image is not None:
                # 2. 드론 물리 정보 가져오기 (HUD용)
                d_vel, _ = p.getBaseVelocity(drone_id, physicsClientId=client_id)
                _, d_orn = p.getBasePositionAndOrientation(drone_id, physicsClientId=client_id)
                d_yaw = p.getEulerFromQuaternion(d_orn)[2]

                # 3. 타겟 상대 속도 계산 (World Frame)
                # (true_rel_pos는 이미 [Step 0]에서 계산됨)
                # 실제 속도 차이 계산 (Target V - Drone V)
                # 타겟 속도는 moving_car_test의 내부 변수(speed, angle)를 알아야 정확함.
                # 여기서는 근사적으로 이전 위치와의 차이로 계산하거나,
                # wrapper에서 정보를 가져와야 함.
                # (간단히 드론 속도의 반대 방향을 표시하는 것으로 대체 가능)
                rel_vel_world = -np.array(d_vel) # 임시: 드론 속도의 반대

                # 4. HUD 그리기
                frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
                
                # conf_score는 Ground Truth 기반이라 없음 (None 처리)
                frame = draw_4way_hud(frame, rel_vel_world, d_yaw, conf_score=None)
                
                # 상태(FSM) 표시
                status_text = "TRACKING" if current_state == STATE_TRACKING else "SEARCHING"
                cv2.putText(frame, status_text, (256-50, 256+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 5. 화면 출력
                cv2.imshow("Drone HUD View", frame)
                if cv2.waitKey(1) == ord('q'):
                    done = True # q 누르면 종료

            # -----------------------------------------------------------
            # [Step 4] 시각화 (Camera Follow)
            # -----------------------------------------------------------
            # 드론 위치 다시 가져오기 (step 이후)
            cur_drone_pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=client_id)
            
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=0,
                cameraPitch=-70,
                cameraTargetPosition=cur_drone_pos,
                physicsClientId=client_id
            )

            sync(steps, start_time, TIMESTEP)

            if done_flag:
                break

        print(f"[Episode {ep+1}] Total Reward: {total_reward:.2f}, Steps: {steps}")

    vec_env.close()
    print("테스트 종료.")