import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym_pybullet_drones.utils.utils import sync

# 기존 프로젝트 모듈 임포트
from moving_car_test import make_custom_env
from learn3 import SiameseTrainingWrapper 
from check import preprocess_rgb, map_heatmap_to_search_linear 

# =============================================================================
# 1. 경로 및 설정
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(current_dir, "results_learn3") 

MODEL_PATH = os.path.join(OUTPUT_FOLDER, "best_model.zip")
STATS_PATH = os.path.join(OUTPUT_FOLDER, "final_vecnormalize.pkl")
SIAMESE_MODEL_PATH = os.path.join(current_dir, "BaselinePretrained.pth.tar")
TEMPLATE_PATH = os.path.join(current_dir, "tracking_object.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# [핵심] 시각화를 위해 내부 정보를 저장하는 커스텀 래퍼
# =============================================================================
class VisualizingSiameseWrapper(SiameseTrainingWrapper):
    def __init__(self, env, model_path, template_path):
        super().__init__(env, model_path, template_path)
        self.vis_info = {
            "image": None,
            "pred_x": None,
            "pred_y": None,
            "heatmap_max": 0.0,
            "rel_vel": np.zeros(3)
        }
        
        # [수정 1] 필터 파라미터 튜닝
        # alpha가 작을수록 반응은 느려지지만 부드러워집니다. (기존 값보다 낮게 설정 추천: 0.1 ~ 0.3)
        self.alpha = 0.3
        
        # [수정 2] 좌표 스무딩용 변수 추가
        self.prev_sx = None
        self.prev_sy = None

    def _get_vision_obs(self):
        # ---------------------------------------------------------------------------
        # 1. 이미지 촬영 및 전처리 (기존 유지)
        # ---------------------------------------------------------------------------
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        rpy = p.getEulerFromQuaternion(orn)
        yaw = rpy[2]

        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R_yaw = np.array([[cos_y, -sin_y, 0], [sin_y,  cos_y, 0], [0, 0, 1]])

        # 카메라 위치 계산 (소프트웨어 짐벌 효과)
        eye_pos = np.array(pos) + np.array([0, 0, -0.05])
        target_pos = eye_pos + np.array([0, 0, -1.0])
        up_vec = R_yaw @ np.array([0, 1, 0])

        view_matrix = p.computeViewMatrix(eye_pos.tolist(), target_pos.tolist(), up_vec.tolist())
        proj_matrix = p.computeProjectionMatrixFOV(fov=90.0, aspect=1.0, nearVal=0.02, farVal=20.0)
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=255, height=255,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client
        )
        
        rgb_image = np.reshape(rgb, (255, 255, 4)).astype(np.uint8)[:, :, :3]

        # ---------------------------------------------------------------------------
        # 2. 샴 네트워크 추론 (기존 유지)
        # ---------------------------------------------------------------------------
        search_tensor = preprocess_rgb(rgb_image, (255,255), self.device, imagenet_norm=False)
        with torch.no_grad():
            resp = self.siam_model(self.template_tensor, search_tensor)
        
        resp = resp.squeeze().detach().cpu().numpy()
        py_hm, px_hm = np.unravel_index(np.argmax(resp), resp.shape)
        sx, sy = map_heatmap_to_search_linear(
            px_hm, py_hm,
            4.255573, 58.647, 3.910728, 63.602,
            half_pixel=True
        )

        # ---------------------------------------------------------------------------
        # 3. 좌표 스무딩 (기존 유지)
        # ---------------------------------------------------------------------------
        if self.prev_sx is not None:
            smooth_factor = 0.4 
            sx = smooth_factor * self.prev_sx + (1 - smooth_factor) * sx
            sy = smooth_factor * self.prev_sy + (1 - smooth_factor) * sy
        
        self.prev_sx = sx
        self.prev_sy = sy

        resp_flat = resp.reshape(-1)
        max_v = resp_flat.max()
        mean_v = resp_flat.mean()
        std_v = resp_flat.std() + 1e-6
        conf = (max_v - mean_v) / std_v

        # ---------------------------------------------------------------------------
        # [수정 핵심 1] 물리 거리 변환 (Z 높이 보정 추가)
        # ---------------------------------------------------------------------------
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        current_z = drone_pos[2]
        
        # 타겟 높이 (moving_car_test.py와 동일하게)
        TARGET_HEIGHT = 0.2  
        rel_height = max(0.1, current_z - TARGET_HEIGHT)

        ground_width_m = 2 * rel_height * np.tan(self.V_FOV_RAD / 2)
        meters_per_pixel = ground_width_m / 255.0
        center_pixel = 255.0 / 2.0
        
        # 이미지(Body) 좌표 계산
        error_y_pixels = sy - center_pixel
        rel_body_x = -error_y_pixels * meters_per_pixel 
        
        error_x_pixels = sx - center_pixel
        rel_body_y = -error_x_pixels * meters_per_pixel 

        # ---------------------------------------------------------------------------
        # [수정 핵심 2] Body Frame -> World Frame 회전 변환
        # ---------------------------------------------------------------------------
        # 드론의 현재 Yaw (위에서 구한 rpy[2] 사용)
        # World_X = Body_X * cos(yaw) - Body_Y * sin(yaw)
        # World_Y = Body_X * sin(yaw) + Body_Y * cos(yaw)
        rel_x_world = rel_body_x * np.cos(yaw) - rel_body_y * np.sin(yaw)
        rel_y_world = rel_body_x * np.sin(yaw) + rel_body_y * np.cos(yaw)

        current_rel_pos = np.array([rel_x_world, rel_y_world, 0.0 - current_z], dtype=np.float32)
        
        # ---------------------------------------------------------------------------
        # 5. 속도 계산 및 시각화 데이터 저장
        # ---------------------------------------------------------------------------
        raw_vel = (current_rel_pos - self.prev_rel_pos) / self.dt
        self.filtered_rel_vel = (self.alpha * raw_vel) + ((1 - self.alpha) * self.filtered_rel_vel)
        
        if conf < 2.5: # 2.5 or 3.0
            final_vel = np.zeros(3, dtype=np.float32)
        else:
            if np.linalg.norm(self.filtered_rel_vel) < 0.05:
                self.filtered_rel_vel = np.zeros(3)
            final_vel = np.clip(self.filtered_rel_vel, -4.0, 4.0)

        self.prev_rel_pos = current_rel_pos 

        # [시각화 데이터 업데이트]
        self.vis_info["image"] = rgb_image.copy() 
        self.vis_info["pred_x"] = sx              
        self.vis_info["pred_y"] = sy              
        self.vis_info["heatmap_max"] = np.max(conf)
        self.vis_info["rel_vel"] = final_vel 

        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        rpy_drone = p.getEulerFromQuaternion(drone_orn)
        prev_action = getattr(self.env, 'prev_action', np.zeros(4, dtype=np.float32))

        obs_vector = np.concatenate([
            current_rel_pos, final_vel, ang_vel, rpy_drone, prev_action
        ]).astype(np.float32)

        return obs_vector

def draw_velocity_arrow(img, rel_vel, scale=50):
    """
    img: 카메라 이미지 프레임 (numpy array)
    rel_vel: 상대 속도 벡터 (예: [vx, vy, vz])
    scale: 속도 값을 픽셀 길이로 변환할 배율 (속도가 작으면 값을 키우세요)
    """
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # rel_vel의 성분 매핑 (사용하는 좌표계에 따라 인덱스 조정 필요)
    # 예: 드론의 경우 보통 x=좌우, z=상하, y=전후
    # 여기서는 x성분을 가로, z성분(혹은 y)을 세로로 가정합니다.
    vx = rel_vel[0]
    vy = rel_vel[1] # 상황에 따라 rel_vel[2]를 사용할 수도 있음
    
    # 끝점 계산 (화면 좌표계: 아래쪽이 y 증가 방향이므로 -vy를 적용할 수 있음)
    end_x = int(center_x + (vx * scale))
    end_y = int(center_y - (vy * scale)) # 위쪽이 +라면 마이너스 처리
    
    # 1. 화살표 그리기
    # 색상: (0, 0, 255) -> 빨간색 (BGR)
    # tipLength: 화살표 머리 크기 비율
    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (0, 0, 255), 3, tipLength=0.3)
    
    # 2. 텍스트 표시 (속도 값)
    label = f"Vx:{vx:.2f}, Vy:{vy:.2f}"
    cv2.putText(img, label, (end_x + 10, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return img

def draw_4way_hud(img, rel_vel_world, drone_yaw, conf_score):
    """
    img: 카메라 이미지
    rel_vel_world: 타겟의 상대 속도(월드 좌표)
    drone_yaw: 드론의 현재 헤딩 각도
    conf_score: 샴 네트워크의 확신도 점수 (float)
    """
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # ---------------------------------------------------------
    # 1. 좌표 변환 (기존 동일)
    # ---------------------------------------------------------
    vx_world, vy_world = rel_vel_world[0], rel_vel_world[1]
    vel_forward = vx_world * np.cos(drone_yaw) + vy_world * np.sin(drone_yaw)
    vel_right = -vx_world * np.sin(drone_yaw) + vy_world * np.cos(drone_yaw)

    # ---------------------------------------------------------
    # 2. 4방향 바 그리기 (기존 동일)
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
    # 3. [추가됨] Conf 점수 및 상태 표시
    # ---------------------------------------------------------
    # Conf 점수에 따른 색상 변화
    if conf_score > 3.0:
        conf_color = (0, 255, 0) # Green (안전)
    elif conf_score > 2.0:
        conf_color = (0, 255, 255) # Yellow (주의)
    else:
        conf_color = (0, 0, 255) # Red (위험)

    # 좌측 상단에 Conf 표시
    cv2.putText(img, f"CONF: {conf_score:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2)

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
# 메인 함수
# =============================================================================
def main():
    print(f"[INFO] 테스트 설정을 시작합니다. Device: {DEVICE}")

    # 1. 환경 생성
    env = DummyVecEnv([lambda: VisualizingSiameseWrapper(
        make_custom_env(gui=True, obs_mode="rgb", is_test_mode=True), 
        SIAMESE_MODEL_PATH,
        TEMPLATE_PATH
    )])

    # 2. 정규화 통계 로드
    if os.path.exists(STATS_PATH):
        print(f"[INFO] 정규화 통계 로드: {STATS_PATH}")
        env = VecNormalize.load(STATS_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[WARNING] 통계 파일 없음!")

    # 3. 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일이 없습니다: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH, env=env, device=DEVICE)
    print(f"[INFO] 모델 로드 완료. 테스트 시작!")

    # -------------------------------------------------------------
    # 래퍼 접근
    # -------------------------------------------------------------
    # env (VecNormalize) -> envs[0] (VisualizingSiameseWrapper)
    vis_wrapper = env.envs[0] 
    
    # 물리 정보 접근
    raw_env = vis_wrapper.env.unwrapped
    
    TIMESTEP = raw_env.CTRL_TIMESTEP
    CLIENT = raw_env.CLIENT
    DRONE_ID = raw_env.DRONE_IDS[0]

    obs = env.reset()
    
    # PyBullet 카메라 초기화
    p.resetDebugVisualizerCamera(3.0, 0, -30, [0, 0, 0], physicsClientId=CLIENT)
    
    for episode in range(10):
        total_reward = 0
        steps = 0
        ep_start_time = time.time()
        obs = env.reset()

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            steps += 1
            
            # =============================================================
            # [시각화]
            # =============================================================
            try:
                vis_data = vis_wrapper.vis_info
                
                if vis_data["image"] is not None:
                    # 1. OpenCV용 이미지 (BGR)
                    # vis_data["image"]는 위에서 uint8로 변환되었으므로 이제 에러 없음
                    frame = cv2.cvtColor(vis_data["image"], cv2.COLOR_RGB2BGR)
                    
                    # 2. 화면 확대 (255x255 -> 512x512)
                    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
                    
                    # 3. 좌표 변환
                    scale = 512 / 255.0
                    pred_x = int(vis_data["pred_x"] * scale)
                    pred_y = int(vis_data["pred_y"] * scale)
                    center_x = 512 // 2
                    center_y = 512 // 2

                    # 4. 그리기
                    # (A) 중앙 십자선 (초록색)
                    cv2.line(frame, (center_x, 0), (center_x, 512), (0, 255, 0), 1)
                    cv2.line(frame, (0, center_y), (512, center_y), (0, 255, 0), 1)
                    
                    # (B) 예측된 타겟 위치 (빨간색 원)
                    cv2.circle(frame, (pred_x, pred_y), 10, (0, 0, 255), 2)
                    cv2.circle(frame, (pred_x, pred_y), 2, (0, 0, 255), -1)
                    
                    # (C) 텍스트 정보
                    score = vis_data["heatmap_max"]
                    #cv2.putText(frame, f"Conf: {score:.2f}", (10, 30), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    # 드론 물리 정보 가져오기 (시각화용)
                    drone_lin_vel, _ = p.getBaseVelocity(DRONE_ID, physicsClientId=CLIENT)
                    _, drone_orn = p.getBasePositionAndOrientation(DRONE_ID, physicsClientId=CLIENT)
                    drone_rpy = p.getEulerFromQuaternion(drone_orn)
                    drone_yaw = drone_rpy[2]
                    rel_vel = vis_data.get("rel_vel")
                    if rel_vel is not None:
                        # 화살표 그리기 (scale 값은 속도 크기에 맞춰 조절: 예 50~100)
                        #frame = draw_velocity_arrow(frame, rel_vel, scale=100)
                        frame = draw_4way_hud(frame, rel_vel, drone_yaw, score)

                    cv2.imshow("Drone Brain View (Red=Pred, Green=Center)", frame)
                
                    if cv2.waitKey(1) == ord('q'):
                        break
            except Exception as e:
                # 에러가 나면 콘솔에 출력하고 계속 진행 (멈추지 않음)
                print(f"Error: {e}")
                pass
            
            # =============================================================

            sync(steps, ep_start_time, TIMESTEP)
            
            if DRONE_ID >= 0:
                drone_pos, _ = p.getBasePositionAndOrientation(DRONE_ID, physicsClientId=CLIENT)
                p.resetDebugVisualizerCamera(3.0, 0, -30, drone_pos, physicsClientId=CLIENT)

            if dones[0]:
                info = infos[0]
                term_reason = info.get('termination_reason', 'unknown')
                print(f"Ep {episode+1} End. Step: {steps}, Rw: {total_reward:.1f}, Reason: {term_reason}")
                break

    env.close()
    cv2.destroyAllWindows()
    print("[INFO] 테스트 종료.")

if __name__ == "__main__":
    main()