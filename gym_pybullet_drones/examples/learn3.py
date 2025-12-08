import os
import time
from datetime import datetime
import numpy as np
import torch
import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback # EvalCallback 등과 함께 있음
from collections import Counter

# 기존 프로젝트 파일들 임포트
from moving_car_test import make_custom_env, MovingTargetWrapper
from model import SiameseNet, BaselineEmbeddingNet
from check import preprocess_rgb, letterbox_rgb, map_heatmap_to_search_linear, TEMPLATE_SIZE

# ---------------- 설 정 ----------------
# [주의] 샴 네트워크를 여러 프로세스에서 띄우면 VRAM을 많이 씁니다.
# VRAM이 부족하면 num_envs를 1 또는 2로 줄이세요.
NUM_ENVS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_TIMESTEPS = 5_000_000

# ---------------- Vision Training Wrapper ----------------
class SiameseTrainingWrapper(gym.Wrapper):
    """
    학습 중에 Siamese Network를 직접 실행하여 Observation을 생성하는 래퍼
    + 위치 차분(Difference)을 이용해 상대 속도(rel_vel)를 추정함
    """
    def __init__(self, env, model_path, template_path):
        super().__init__(env)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 샴 네트워크 로드
        self.embedding_net = BaselineEmbeddingNet().to(self.device)
        self.siam_model = SiameseNet(self.embedding_net).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        sd = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.siam_model.load_state_dict(sd, strict=True)
        self.siam_model.eval() # 학습 모드 끄기 (추론만)

        # 2. 템플릿 이미지 로드
        template_img_bgr = cv2.imread(template_path)
        if template_img_bgr is None:
            raise FileNotFoundError(f"템플릿 이미지를 찾을 수 없습니다: {template_path}")
        template_img_rgb = cv2.cvtColor(template_img_bgr, cv2.COLOR_BGR2RGB)
        self.template_tensor = preprocess_rgb(template_img_rgb, TEMPLATE_SIZE, self.device, imagenet_norm=False)

        # 3. Observation Space 정의 (16차원)
        # [rel_pos(3), rel_vel(3), ang_vel(3), rpy(3), prev_action(4)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # 4. 속도 계산을 위한 변수
        self.prev_rel_pos = np.zeros(3, dtype=np.float32)
        self.dt = float(self.env.unwrapped.CTRL_TIMESTEP)
        
        # 카메라 파라미터 (wrapper.py와 동일하게 설정)
        self.CAM_H_RES = 255.0
        self.CAM_V_RES = 255.0
        self.CAM_CENTER_X = self.CAM_H_RES / 2.0
        self.CAM_CENTER_Y = self.CAM_V_RES / 2.0
        self.V_FOV_RAD = np.deg2rad(90.0)  #60 -> 90
        self.H_FOV_RAD = 2 * np.arctan(np.tan(self.V_FOV_RAD / 2) * (self.CAM_H_RES / self.CAM_V_RES))
        #self.H_FOV_RAD = self.V_FOV_RAD #정사각형이므로 수직, 수평 시야각 동일

        self.drone_id = self.env.unwrapped.DRONE_IDS[0]
        self.client = self.env.unwrapped.CLIENT

        # [추가] 필터링을 위한 이전 속도 저장 변수
        self.filtered_rel_vel = np.zeros(3, dtype=np.float32)
        
        # 필터 강도 (0.0 ~ 1.0): 클수록 부드럽지만 반응이 느림
        # 0.1 ~ 0.2 정도 추천
        self.alpha = 0.3
        # [수정 2] 좌표 스무딩용 변수 초기화
        self.prev_sx = None
        self.prev_sy = None

    def reset(self, **kwargs):
        # 환경 리셋 (moving_car_test의 reset은 obs를 반환하므로 받아둠)
        _, info = self.env.reset(**kwargs)

        # [수정 3] 리셋 시 스무딩 변수도 초기화
        self.prev_rel_pos = np.zeros(3, dtype=np.float32)
        self.filtered_rel_vel = np.zeros(3, dtype=np.float32)
        self.prev_sx = None
        self.prev_sy = None
        
        # 속도 계산용 이전 위치 초기화
        self.prev_rel_pos = np.zeros(3, dtype=np.float32)

        self.filtered_rel_vel = np.zeros(3, dtype=np.float32)
        
        # 첫 프레임 처리
        vision_obs = self._get_vision_obs()
        return vision_obs, info

    def step(self, action):
        # 환경 스텝 (여기서 반환되는 obs는 무시하고 비전으로 새로 만듦)
        _, reward, terminated, truncated, info = self.env.step(action)
        
        # 비전 처리 및 속도 계산
        vision_obs = self._get_vision_obs()
        
        return vision_obs, reward, terminated, truncated, info

    def _get_vision_obs(self):
        import pybullet as p
        import numpy as np

        # ---------------------------------------------------------------------------
        # 1. 255x255 해상도로 직접 촬영
        # ---------------------------------------------------------------------------
        # pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        # R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # eye_pos = np.array(pos) + (R @ np.array([0, 0, -0.05]))
        # target_pos = eye_pos + (R @ np.array([0, 0, -1]))
        # up_vec = (R @ np.array([0, 1, 0]))

        # [수정 후] 소프트웨어 짐벌 적용 (Yaw만 반영, Pitch/Roll 무시)
        pos, orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        
        # 1. 쿼터니언에서 오일러 각도 추출 (Roll, Pitch, Yaw)
        rpy = p.getEulerFromQuaternion(orn)
        yaw = rpy[2] # 헤딩(Yaw)만 가져옴

        # 2. Yaw 회전 행렬만 새로 생성 (Z축 회전)
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        R_yaw = np.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1]
        ])

        # 3. 카메라 위치 및 방향 설정 (기울기 무시)
        # eye_pos: 드론 위치에서 수직 아래로 5cm (드론이 기울어도 카메라는 수평 유지)
        eye_pos = np.array(pos) + np.array([0, 0, -0.05])
        
        # target_pos: 카메라에서 수직 아래 (항상 바닥을 정면으로 봄)
        target_pos = eye_pos + np.array([0, 0, -1.0])
        
        # up_vec: 이미지의 '위쪽'이 드론의 진행 방향(Yaw)을 따라가도록 설정
        # (기존 코드가 Body Y축([0,1,0])을 up으로 썼으므로, Yaw 회전된 Y축 사용)
        up_vec = R_yaw @ np.array([0, 1, 0])

        view_matrix = p.computeViewMatrix(eye_pos.tolist(), target_pos.tolist(), up_vec.tolist())

        # [중요] aspect ratio를 1.0 (255/255)으로 설정
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=90.0, 
            aspect=1.0, 
            nearVal=0.02,
            farVal=20.0
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=255,   # [수정] 320 -> 255
            height=255,  # [수정] 240 -> 255
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client
        )

        rgb_image = np.reshape(rgb, (255, 255, 4))[:, :, :3]

        # ---------------------------------------------------------------------------
        # 2. 전처리 (Letterbox 삭제!)
        # ---------------------------------------------------------------------------
        # [삭제됨] search_rgb_letterboxed, scale, pad_x, pad_y = letterbox_rgb(...)
        # 이미지가 이미 255x255이므로 바로 텐서로 변환합니다.
        search_tensor = preprocess_rgb(rgb_image, (255,255), self.device, imagenet_norm=False)
        
        with torch.no_grad():
            resp = self.siam_model(self.template_tensor, search_tensor)
        
        # ---------------------------------------------------------------------------
        # 3. 좌표 변환 (복잡한 역변환 삭제!)
        # ---------------------------------------------------------------------------
        resp = resp.squeeze().detach().cpu().numpy()
        py_hm, px_hm = np.unravel_index(np.argmax(resp), resp.shape)
        
        # 히트맵 -> 255 이미지 좌표 변환 (이건 샴 네트워크 특성이라 유지해야 함)
        sx, sy = map_heatmap_to_search_linear(
            px_hm, py_hm,
            4.255573, 58.647, 3.910728, 63.602,
            half_pixel=True
        )
        # [수정 4] 좌표 스무딩 적용 (test3.py와 동일 로직)
        if self.prev_sx is not None:
            smooth_factor = 0.4 # 값이 클수록 이전 값 비중이 높음 (부드러움)
            sx = smooth_factor * self.prev_sx + (1 - smooth_factor) * sx
            sy = smooth_factor * self.prev_sy + (1 - smooth_factor) * sy
        self.prev_sx = sx
        self.prev_sy = sy
        # [삭제됨] sx_orig = (sx - pad_x) / scale
        # 이미지가 원본 그 자체이므로 변환된 좌표가 곧 원본 좌표입니다.
        sx_orig = sx
        sy_orig = sy

        resp_flat = resp.reshape(-1)
        max_v = resp_flat.max()
        mean_v = resp_flat.mean()
        std_v = resp_flat.std() + 1e-6
        conf = (max_v - mean_v) / std_v

        # ---------------------------------------------------------------------------
        # 4. 물리 거리 변환 (픽셀 -> 미터)
        # ---------------------------------------------------------------------------
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        current_z = drone_pos[2]
        
        # 타겟의 대략적인 높이 (환경 설정에 따라 0.2 ~ 0.5 등)
        # moving_car_test.py의 init_target_z 값을 참고하여 설정하세요.
        TARGET_HEIGHT = 0.2  
        
        # 카메라와 타겟 사이의 수직 거리 (Depth)
        rel_height = max(0.1, current_z - TARGET_HEIGHT)

        # (255 해상도 기준 미터 변환)
        ground_width_m = 2 * rel_height * np.tan(self.V_FOV_RAD / 2)
        meters_per_pixel = ground_width_m / 255.0

        # 중심점 (127.5)
        center_pixel = 255.0 / 2.0
        
        # [이미지 좌표계 -> 바디 좌표계]
        # 이미지: X(우), Y(하)
        # 바디: X(전방), Y(좌측)
        # 이미지 상단(Y=0)에 있으면 -> error_y < 0 -> 드론 기준 전방(+) -> rel_body_x > 0
        error_y_pixels = sy_orig - center_pixel
        rel_body_x = -error_y_pixels * meters_per_pixel 
        
        # 이미지 우측(X=255)에 있으면 -> error_x > 0 -> 드론 기준 우측 -> rel_body_y < 0 (좌측이 +Y라면)
        error_x_pixels = sx_orig - center_pixel
        rel_body_y = -error_x_pixels * meters_per_pixel

        # ---------------------------------------------------------------------------
        # [수정 핵심 1] Body Frame -> World Frame 좌표 변환
        # ---------------------------------------------------------------------------
        # 드론의 현재 Yaw 각도 구하기
        rpy = p.getEulerFromQuaternion(drone_orn)
        yaw = rpy[2]

        # 2D 회전 행렬 적용
        # World_X = Body_X * cos(yaw) - Body_Y * sin(yaw)
        # World_Y = Body_X * sin(yaw) + Body_Y * cos(yaw)
        rel_x_world = rel_body_x * np.cos(yaw) - rel_body_y * np.sin(yaw)
        rel_y_world = rel_body_x * np.sin(yaw) + rel_body_y * np.cos(yaw)

        # 최종 상대 좌표 (World Frame)
        current_rel_pos = np.array([rel_x_world, rel_y_world, 0.0 - current_z], dtype=np.float32)

        # ---------------------------------------------------------------------------
        # 5. 속도 계산 및 반환 (이전과 동일)
        # ---------------------------------------------------------------------------
        raw_vel = (current_rel_pos - self.prev_rel_pos) / self.dt
        self.filtered_rel_vel = (self.alpha * raw_vel) + ((1 - self.alpha) * self.filtered_rel_vel) #상대속도에 Low-Pass Filter 적용
        # [추가] 신뢰도가 너무 낮으면(타겟 놓침) 속도 정보를 0으로 죽여버림
        # 이렇게 하면 드론이 엉뚱한 곳으로 급발진하는 것을 막음
        if conf < 2.5:  # 기준값은 상황에 맞춰 조정 (2.5 ~ 3.0)
            final_vel = np.zeros(3, dtype=np.float32)
        # 너무 작은 속도는 0으로 처리 (노이즈 제거용 데드존)
        else:
            if np.linalg.norm(self.filtered_rel_vel) < 0.05:
                self.filtered_rel_vel = np.zeros(3)
            final_vel = np.clip(self.filtered_rel_vel, -4.0, 4.0)
        #final_vel = self.filtered_rel_vel.copy()
        self.prev_rel_pos = current_rel_pos 

        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        rpy = p.getEulerFromQuaternion(drone_orn)
        prev_action = getattr(self.env, 'prev_action', np.zeros(4, dtype=np.float32))

        obs_vector = np.concatenate([
            current_rel_pos, final_vel, ang_vel, rpy, prev_action
        ]).astype(np.float32)

        return obs_vector
    

class TerminationStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.counter = Counter()

    def _on_step(self) -> bool:
        # VecEnv라 dones, infos는 벡터 형태
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done:
                    reason = info.get("termination_reason", None)
                    if reason is not None:
                        self.counter[reason] += 1
        return True

    def _on_rollout_end(self) -> None:
            # rollout 한 번 끝날 때마다 TensorBoard에 기록
            total = sum(self.counter.values()) or 1
            for reason, cnt in self.counter.items():
                # 절대 횟수
                self.logger.record(f"term_count/{reason}", float(cnt))
                # 비율(%)도 보고 싶으면
                self.logger.record(f"term_ratio/{reason}", float(cnt) / total)

            # 다음 rollout을 위해 초기화
            self.counter.clear()

# ---------------- 메인 학습 코드 ----------------
if __name__ == "__main__":
    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    SIAMESE_MODEL_PATH = os.path.join(current_dir, "BaselinePretrained.pth.tar")
    TEMPLATE_PATH = os.path.join(current_dir, "tracking_object.png")
    OUTPUT_FOLDER = os.path.join(current_dir, "results_learn3")

    # [추가] learn2의 결과물 경로 (폴더명이 다르면 수정하세요!)
    LEARN2_FOLDER = os.path.join(current_dir, "results_learn2") # learn2 결과 폴더
    STATS_PATH = os.path.join(LEARN2_FOLDER, "final_vecnormalize.pkl")
    MODEL_PATH = os.path.join(LEARN2_FOLDER, "final_model.zip")

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 2. 환경 생성 함수 정의 (기존 동일)
    def make_env(rank: int, seed: int = 0):
        def _init():
            env = make_custom_env(gui=False, obs_mode="rgb", is_test_mode=False)
            env.reset(seed=seed + rank)
            env.unwrapped.IMG_RES = np.array([255, 255])
            env = SiameseTrainingWrapper(env, SIAMESE_MODEL_PATH, TEMPLATE_PATH)
            return env
        return _init

    print(f"[INFO] {NUM_ENVS}개의 환경에서 학습을 시작합니다. (Device: {DEVICE})")
    
    # 3. Train Env 생성 및 통계 로드 (수정됨)
    train_env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
    train_env = VecMonitor(train_env)

    # [핵심 1] learn2의 통계 파일(VecNormalize)이 있으면 불러오기
    if os.path.exists(STATS_PATH):
        print(f"[INFO] learn2의 정규화 통계(VecNormalize)를 로드합니다: {STATS_PATH}")
        # 파일에서 통계(평균, 분산)를 불러와서 환경에 적용
        train_env = VecNormalize.load(STATS_PATH, train_env)
        # 중요: 새로운 환경(비전)에 적응해야 하므로 통계 업데이트는 켭니다 (training=True)
        #fine-tuning에서는 training=False
        train_env.training = True 
        train_env.norm_reward = False 
    else:
        print(f"[WARNING] 통계 파일({STATS_PATH})이 없습니다. 맨땅에서 시작합니다.")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 4. Eval Env 생성 (수정됨)
    eval_env = SubprocVecEnv([make_env(999)])
    eval_env = VecMonitor(eval_env)
    # Eval 환경도 로드된 train_env의 설정을 따라야 하므로 동기화
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)
    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms
    
    # 5. 모델 로드 또는 생성 (수정됨)
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] learn2의 사전 학습 모델을 로드합니다: {MODEL_PATH}")
        # [핵심 2] 모델 불러오기 + 파인튜닝 설정
        model = PPO.load(
            MODEL_PATH,
            env=train_env,
            device=DEVICE, # 기존 코드에 cpu로 되어있던데, 가급적 cuda(DEVICE) 추천
            batch_size=2048,
            n_steps=2048,
            n_epochs=6,
            custom_objects={
                "learning_rate": 5e-5,
                "ent_coef": 0.01,
                "clip_range": 0.1,
                "tensorboard_log": os.path.join(OUTPUT_FOLDER, "tb_logs")
            }
        )
    else:
        print(f"[INFO] 모델 파일({MODEL_PATH})이 없습니다. 처음부터 학습합니다.")
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=os.path.join(OUTPUT_FOLDER, "tb_logs"),
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=1024,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            device=DEVICE # cpu보다는 cuda가 빠릅니다
        )

    # 6. 콜백 및 학습 시작 (기존 동일)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=OUTPUT_FOLDER,
        log_path=OUTPUT_FOLDER,
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // NUM_ENVS, 1),
        save_path=OUTPUT_FOLDER,
        name_prefix="ppo_siamese",
        save_vecnormalize=True
    )

    termination_stats_cb = TerminationStatsCallback()

    callback_list = [eval_callback, checkpoint_callback, termination_stats_cb]

    print("[INFO] 학습 시작...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_list)
    
    # 최종 저장
    model.save(os.path.join(OUTPUT_FOLDER, "final_model"))
    train_env.save(os.path.join(OUTPUT_FOLDER, "final_vecnormalize.pkl"))
    print("[INFO] 학습 완료.")