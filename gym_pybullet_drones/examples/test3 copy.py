import os
import time
import numpy as np
import torch
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
    """
    learn3.py의 래퍼를 상속받아, 예측된 좌표(sx, sy)와 원본 이미지를
    self.vis_info에 저장하여 외부에서 꺼내 쓸 수 있게 만든 클래스
    """
    def __init__(self, env, model_path, template_path):
        super().__init__(env, model_path, template_path)
        self.vis_info = {
            "image": None,
            "pred_x": None,
            "pred_y": None,
            "heatmap_max": 0.0
        }

    def _get_vision_obs(self):
        # 1. 이미지 촬영
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        eye_pos = np.array(pos) + (R @ np.array([0, 0, -0.05]))
        target_pos = eye_pos + (R @ np.array([0, 0, -1]))
        up_vec = (R @ np.array([0, 1, 0]))
        view_matrix = p.computeViewMatrix(eye_pos.tolist(), target_pos.tolist(), up_vec.tolist())
        proj_matrix = p.computeProjectionMatrixFOV(fov=90.0, aspect=1.0, nearVal=0.02, farVal=20.0)
        
        _, _, rgb, _, _ = p.getCameraImage(
            width=255, height=255,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client
        )
        
        # [!!! 수정된 부분 !!!] .astype(np.uint8) 추가 -> int32를 uint8로 변환하여 OpenCV 에러 해결
        rgb_image = np.reshape(rgb, (255, 255, 4)).astype(np.uint8)[:, :, :3]

        # 2. 샴 네트워크 추론
        search_tensor = preprocess_rgb(rgb_image, (255,255), self.device, imagenet_norm=False)
        with torch.no_grad():
            resp = self.siam_model(self.template_tensor, search_tensor)
        
        # 3. 좌표 계산
        resp = resp.squeeze().detach().cpu().numpy()
        py_hm, px_hm = np.unravel_index(np.argmax(resp), resp.shape)
        sx, sy = map_heatmap_to_search_linear(
            px_hm, py_hm,
            4.255573, 58.647, 3.910728, 63.602,
            half_pixel=True
        )
        
        # [데이터 저장] 외부 시각화용
        self.vis_info["image"] = rgb_image.copy() 
        self.vis_info["pred_x"] = sx              
        self.vis_info["pred_y"] = sy              
        self.vis_info["heatmap_max"] = np.max(resp)

        # 4. 관측 벡터 생성 (학습 코드와 동일 로직)
        sx_orig, sy_orig = sx, sy
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        current_z = drone_pos[2]
        ground_width_m = 2 * current_z * np.tan(self.V_FOV_RAD / 2)
        meters_per_pixel = ground_width_m / 255.0
        center_pixel = 255.0 / 2.0
        
        error_y_pixels = sy_orig - center_pixel
        rel_x = -error_y_pixels * meters_per_pixel 
        error_x_pixels = sx_orig - center_pixel
        rel_y = -error_x_pixels * meters_per_pixel 

        current_rel_pos = np.array([rel_x, rel_y, 0.0 - current_z], dtype=np.float32)
        
        raw_vel = (current_rel_pos - self.prev_rel_pos) / self.dt
        self.filtered_rel_vel = (self.alpha * raw_vel) + ((1 - self.alpha) * self.filtered_rel_vel)
        final_vel = np.clip(self.filtered_rel_vel, -4.0, 4.0)
        self.prev_rel_pos = current_rel_pos 

        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        rpy = p.getEulerFromQuaternion(drone_orn)
        prev_action = getattr(self.env, 'prev_action', np.zeros(4, dtype=np.float32))

        obs_vector = np.concatenate([
            current_rel_pos, final_vel, ang_vel, rpy, prev_action
        ]).astype(np.float32)

        return obs_vector

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
                    cv2.putText(frame, f"Conf: {score:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    cv2.imshow("Drone Brain View (Red=Pred, Green=Center)", frame)
                
                    if cv2.waitKey(1) == ord('q'):
                        break
            except Exception as e:
                # 에러가 나면 콘솔에 출력하고 계속 진행 (멈추지 않음)
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