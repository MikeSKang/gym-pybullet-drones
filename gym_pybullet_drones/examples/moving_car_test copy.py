import os
import time
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
from gymnasium.wrappers import TimeLimit

# gym-pybullet-drones에서 필요한 클래스들을 직접 import
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType
from gym_pybullet_drones.utils.enums import DroneModel, Physics

"""강화학습할 때 이 커스텀 환경 만들기"""
def make_custom_env(gui=False, obs_mode="rel_pos"):
    INIT_XYZS = np.array([[0, 0, 10.0]])
    base_env = CustomHoverAviary(gui=gui, initial_xyzs=INIT_XYZS, act=ActionType.VEL)
    base_env = TimeLimit(base_env, max_episode_steps=2400)

    env = MovingTargetWrapper(base_env, camera_size=(84, 84), obs_mode=obs_mode)
    env = TextureWrapper(env, "gym_pybullet_drones/examples/textures/floor1.jpg")
    #env = BoundaryWrapper(env, x_max=25.0, y_max=25.0, z_min=1.0, z_max=20.0)
    return env

# -------------------- Custom HoverAviary --------------------
class CustomHoverAviary(HoverAviary):
    def _computeTerminated(self):
        return False  # 성공 조건 무시 (학습 목적이면 직접 정의해도 됨)

    def _computeTruncated(self):
        return False  # 활동 반경 제한 및 시간 제한 무시

# -------------------- BoundaryWrapper --------------------
class BoundaryWrapper(gym.Wrapper):
    def __init__(self, env, x_max=25.0, y_max=25.0, z_min=1.0, z_max=15.0):
        super().__init__(env)
        self.x_max, self.y_max = x_max, y_max
        self.z_min, self.z_max = z_min, z_max

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 드론 위치 확인
        pos, _ = p.getBasePositionAndOrientation(
            self.env.unwrapped.DRONE_IDS[0], physicsClientId=self.env.unwrapped.CLIENT
        )
        x, y, z = pos
        # 소프트 경계 패널티
        margin_xy = max(0.0, abs(x) - 0.8*self.x_max) + max(0.0, abs(y) - 0.8*self.y_max)
        margin_z  = max(0.0, self.z_min - z) + max(0.0, z - self.z_max)
        boundary_penalty = -0.2 * (margin_xy + 2.0*margin_z)

        reward += boundary_penalty

        return obs, reward, terminated, truncated, info

# -------------------- MovingTargetWrapper --------------------
class MovingTargetWrapper(gym.Wrapper):
    """
    - 원본 환경 위에 타겟 오브젝트(큐브)를 추가하고,
    - 사각형 범위 내에서 자동으로 움직이도록 합니다.
    - 드론 시점의 카메라 RGB 이미지를 관측으로 반환합니다.
    """
    def __init__(self, env,
                 obs_mode="rel_pos",      # "rgb"/"rel_pos"/"both"
                 camera_size=(84, 84),
                 fov_deg=90.0, near=0.02, far=20.0,
                 init_target_z=0.25,
                 target_urdf="cube.urdf", target_rgba=(1.0, 0.2, 0.2, 1.0),
                 x_max=50.0, y_max=50.0,  # 타겟 이동 가능 범위(수평)
                 speed=0.01,

                 # ---- 추가: 보상/종료 파라미터 ----
                 desired_alt_above=10.0,  # 타겟 '상공' 목표 고도 차
                 xy_tol=1.5,              # 성공 밴드(수평)
                 z_tol=1.0,               # 성공 밴드(수직)
                 max_xy=30.0,             # 너무 멀어지면 종료
                 min_z=0.2,               # 거의 충돌 수준
                 alive_bonus=0.05,        # 매 스텝 생존 보너스
                 w_xy=5.0,                # 수평 오차 가중치
                 w_z=3.5,                 # 수직 오차 가중치
                 w_act=0.001,             # 액션 크기 패널티
                 w_vel=0.001,             # 드론 속도 패널티(선택)
                 success_reward=2.0,      # 성공 밴드 안이면 추가 보상
                 patience_steps=180       # 일정 시간 너무 멀면 종료
                 ):
        super().__init__(env)
        self.obs_mode = obs_mode
        # 보상/종료 파라미터 저장
        self.desired_alt_above = float(desired_alt_above)
        self.xy_tol = float(xy_tol)
        self.z_tol = float(z_tol)
        self.max_xy = float(max_xy)
        self.min_z = float(min_z)
        self.alive_bonus = float(alive_bonus)
        self.w_xy = float(w_xy)
        self.w_z = float(w_z)
        self.w_act = float(w_act)
        self.w_vel = float(w_vel)
        self.success_reward = float(success_reward)
        self.patience_steps = int(patience_steps)
        # 내부 상태
        self._far_counter = 0
        # PyBullet 리소스 경로
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # 타겟 관련 설정
        self.init_target_z = float(init_target_z)
        self._target_id = None
        self._target_urdf = target_urdf
        self._target_rgba = target_rgba
        self._client = getattr(self.env.unwrapped, "CLIENT", 0)

        # 드론 핸들
        drone_ids = getattr(self.env.unwrapped, "DRONE_IDS", None)
        self._drone_id = drone_ids[0] if drone_ids else None

        # 카메라 설정
        self.W, self.H = int(camera_size[0]), int(camera_size[1])
        self.fov = fov_deg
        aspect = self.W / float(self.H)
        self._proj = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=aspect, nearVal=near, farVal=far
        )
        self._img_flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX

        # 관측/행동 공간        
        if self.obs_mode == "rgb":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
            )
        elif self.obs_mode == "rel_pos":
            # 상대위치(3) + 선속도(3) + 각속도(3) =9
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
            )
        elif self.obs_mode == "both":
            self.observation_space = spaces.Dict({
                "rgb": spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8),
                # rel_pos 부분의 shape을 9로 변경. 상대위치(3) + 선속도(3) + 각속도(3) = 9
                "rel_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
            })
        self.action_space = self.env.action_space


        # 움직임 관련
        self.max_speed = 2.0        # 자동차의 최대 속도
        self.max_accel = 1.0       # 최대 가속/감속
        self.max_turn_rate = 0.05    # 최대 회전율 (라디안)
        
        self.speed = 0.05            # 현재 속도
        self.angle = np.random.uniform(0, 2 * np.pi) # 현재 진행 방향 (각도)
        self.turn_rate = 0.0         # 현재 회전율
        self.accel = 0.0             # 현재 가속도

        self.x_max, self.y_max = x_max, y_max
        self.target_pos = np.array([0.0, 0.0, self.init_target_z])

    # -------------------- 타겟 관련 --------------------
    def _spawn_or_reset_target(self, pos, orn):
        """타겟 생성 또는 위치 리셋"""
        if self._target_id is None:
            self._target_id = p.loadURDF(
                self._target_urdf,
                pos,
                orn,
                useFixedBase=False,
                physicsClientId=self._client,
                globalScaling=0.5,
            )

            # 텍스처 적용 코드 추가
            texture_path = "gym_pybullet_drones/examples/textures/moving_obj2.png"
            if os.path.exists(texture_path):
                texture_id = p.loadTexture(texture_path)
                if texture_id != -1:
                    p.changeVisualShape(
                        self._target_id,
                        -1,
                        textureUniqueId=texture_id,
                        physicsClientId=self._client
                    )
            else:
                # 텍스처 없으면 색상 적용
                num_vis = p.getVisualShapeData(self._target_id, physicsClientId=self._client)
                for v in num_vis:
                    link_idx = v[1]
                    p.changeVisualShape(
                        self._target_id,
                        link_idx,
                        rgbaColor=self._target_rgba,
                        physicsClientId=self._client,
                    )

        else:
            p.resetBasePositionAndOrientation(
                self._target_id, pos, orn, physicsClientId=self._client
            )
            p.resetBaseVelocity(
                self._target_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client
            )
    def _update_target(self):
        """경계 내에서 타겟 위치를 부드럽고 임의적으로 업데이트"""
        # 1. 가속도와 회전율에 작은 임의의 변화를 줍니다.
        self.accel += np.random.uniform(-0.001, 0.001)
        self.turn_rate += np.random.uniform(-0.01, 0.01)
        
        # 2. 가속도와 회전율을 최대값으로 제한합니다.
        self.accel = np.clip(self.accel, -self.max_accel, self.max_accel)
        self.turn_rate = np.clip(self.turn_rate, -self.max_turn_rate, self.max_turn_rate)

        # 3. 현재 속도와 진행 방향 각도를 업데이트합니다.
        self.speed += self.accel
        self.speed = np.clip(self.speed, 0, self.max_speed) # 속도는 0 이상, 최대 속도 이하
        self.angle = (self.angle + self.turn_rate) % (2 * np.pi) # 각도는 0 ~ 2pi

        # 4. 새로운 속도 벡터(vx, vy)를 계산합니다.
        vx = self.speed * np.cos(self.angle)
        vy = self.speed * np.sin(self.angle)

        # 5. 새로운 위치를 계산합니다.
        self.dt = 1.0/240.0
        self.target_pos[0] += vx * self.dt
        self.target_pos[1] += vy * self.dt

        # 6. 경계 처리: 경계에 가까워지면 중앙으로 부드럽게 방향을 틉니다.
        turn_away_strength = 0.0
        if abs(self.target_pos[0]) > self.x_max * 0.9:
            # x 경계 근처 -> y축 방향으로 회전 유도
            target_angle = np.pi / 2 if self.target_pos[1] < 0 else 3 * np.pi / 2
            turn_away_strength = 0.02
        if abs(self.target_pos[1]) > self.y_max * 0.9:
            # y 경계 근처 -> x축 방향으로 회전 유도
            target_angle = np.pi if self.target_pos[0] < 0 else 0
            turn_away_strength = 0.02
        
        if turn_away_strength > 0:
            # 현재 각도와 목표 각도의 차이를 계산하여 회전율에 반영
            angle_diff = (target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi
            self.turn_rate += angle_diff * turn_away_strength
            # 위치를 경계 안으로 강제 조정
            self.target_pos[0] = np.clip(self.target_pos[0], -self.x_max, self.x_max)
            self.target_pos[1] = np.clip(self.target_pos[1], -self.y_max, self.y_max)

        # 7. 최종 위치를 시뮬레이션에 적용합니다.
        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        p.resetBasePositionAndOrientation(
            self._target_id, self.target_pos.tolist(), orn, physicsClientId=self._client
        )

    # -------------------- 카메라 --------------------
    def _view_matrix_from_drone(self):
        """드론 시점 카메라 뷰 매트릭스"""
        pos, q = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        fwd = R @ np.array([0, 0, -1])
        up = R @ np.array([0, 1, 0])
        eye = np.array(pos) + (R @ np.array([0, 0, -0.05]))
        target = eye + fwd
        return p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())

    def _render_rgb(self):
        """드론 시점 카메라 이미지"""
        view = self._view_matrix_from_drone()
        _, _, rgb, _, _ = p.getCameraImage(
            width=self.W,
            height=self.H,
            viewMatrix=view,
            projectionMatrix=self._proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=self._img_flags,
            physicsClientId=self._client,
        )
        rgb = np.reshape(rgb, (self.H, self.W, 4))[:, :, :3]
        return rgb.astype(np.uint8, copy=False)

    # -------------------- Gym API --------------------
    def reset(self, **kwargs):
        obs_base, info = self.env.reset(**kwargs)
        # self.env.reset()으로 시뮬레이션이 초기화되었으므로,
        # 이전 타겟 ID는 더 이상 유효하지 않습니다. None으로 초기화합니다.
        self._target_id = None
        
        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self._spawn_or_reset_target([0.0, 0.0, self.init_target_z], orn)

        # 랜덤 속도 초기화
        self.target_vel = np.random.uniform(-self.speed, self.speed, size=2)
        self.target_pos = np.array([0.0, 0.0, self.init_target_z])

        # 종료/보상 관련 내부 상태 초기화
        self._far_counter = 0

        # 좌표 계산
        drone_pos, _ = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(self._target_id, physicsClientId=self._client)
        rel_pos = np.array(target_pos) - np.array(drone_pos)

        # 자기인식 정보 추가: 드론의 속도와 각속도
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=self._client)
        full_obs_vector = np.concatenate([rel_pos, lin_vel, ang_vel]).astype(np.float32)

        # obs 모드별 반환
        if self.obs_mode == "rgb":
            rgb = self._render_rgb()
            obs = rgb
        elif self.obs_mode == "rel_pos":
            obs = full_obs_vector
        elif self.obs_mode == "both":
            rgb = self._render_rgb()
            obs = {"rgb": rgb, "rel_pos": full_obs_vector}

        return obs, info


    def step(self, action):
        # 1. 에이전트의 행동을 실제 환경에 적용하고, 타겟을 움직입니다.
        obs_base, reward_base, term_base, trunc_base, info = self.env.step(action)
        
        prev_target_pos = self.target_pos.copy()
        prev_drone_pos, _ = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        prev_dist = np.linalg.norm(np.array(prev_target_pos[:2]) - np.array(prev_drone_pos[:2]))
        self._update_target() 

        # 2. 행동 후의 새로운 상태 정보를 가져옵니다.
        drone_pos, drone_quat = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(self._target_id, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=self._client)

        # 3. 보상 및 종료 조건 계산에 필요한 값들을 정의합니다.
        dist_vec = np.array(target_pos) - np.array(drone_pos)
        dist_xy = np.linalg.norm(dist_vec[:2])
        dist_z = dist_vec[2] # Z축 거리 (부호 있음)
        
        # =====================================================================
        # ===== 개선된 보상 함수 로직 =====
        # =====================================================================
        reward = self.alive_bonus # 살아있으면 매 스텝 기본 점수

        curr_dist = np.linalg.norm(np.array(target_pos[:2]) - np.array(drone_pos[:2]))
        reward += (prev_dist - curr_dist) * 3.0   # ← distance 변화 보상 추가 (가중치 3~5)

        # [개선 1] 거리 오차에 대한 페널티 (Gaussian-like)
        # 수평(xy) 거리 오차: 목표는 타겟 바로 위(xy 오차 0)
        xy_error = dist_xy
        reward -= self.w_xy * (1 - np.exp(-0.5 * xy_error**2))

        # 수직(z) 거리 오차: 목표는 타겟보다 desired_alt_above 만큼 위에 있는 것
        z_error = abs((drone_pos[2] - target_pos[2]) - self.desired_alt_above)
        reward -= self.w_z * (1 - np.exp(-0.5 * z_error**2))

        # [개선 2] 드론의 자세(Yaw) 페널티
        # 드론의 전방 벡터 계산
        rot_matrix = np.array(p.getMatrixFromQuaternion(drone_quat)).reshape(3, 3)
        fwd_vec = rot_matrix[:, 0]  # X축이 전방
        
        # 드론의 전방 벡터와 타겟 방향 벡터 사이의 각도 오차 계산
        target_dir_vec = dist_vec / (np.linalg.norm(dist_vec) + 1e-6)
        orientation_error = 1 - np.dot(fwd_vec[:2], target_dir_vec[:2]) # 수평(XY) 평면에서만 고려
        reward -= 0.1 * orientation_error # 가중치 0.1 (조정 가능)

        # [개선 3] 제어 입력(Action) 및 속도 페널티
        reward -= self.w_act * np.linalg.norm(action)
        reward -= self.w_vel * np.linalg.norm(lin_vel)

        # [개선 4] 성공 보너스
        # 수평 및 수직 오차가 허용 범위(tolerance) 안에 들어오면 추가 보상
        is_success = (xy_error < self.xy_tol) and (z_error < self.z_tol)
        if is_success:
            reward += self.success_reward
        
        # =====================================================================

        # 5. 에피소드 종료 조건을 계산합니다.
        # 너무 멀어지면 종료
        if dist_xy > self.max_xy:
            self._far_counter += 1
        else:
            self._far_counter = 0
        
        # 기존 충돌 조건 + 너무 멀리 떨어져 있는 시간이 길어지면 종료
        terminated = bool(term_base or (drone_pos[2] <= self.min_z) or (self._far_counter > self.patience_steps))
        truncated = bool(trunc_base)

        # 6. 에이전트에게 전달할 관측(Observation) 정보를 구성합니다.
        rel_pos = dist_vec
        full_obs_vector = np.concatenate([rel_pos, lin_vel, ang_vel]).astype(np.float32)

        if self.obs_mode == "rgb":
            rgb = self._render_rgb()
            obs = rgb
        elif self.obs_mode == "rel_pos":
            obs = full_obs_vector
        elif self.obs_mode == "both":
            rgb = self._render_rgb()
            obs = {"rgb": rgb, "rel_pos": full_obs_vector}
        else: # 기본값 설정
            obs = full_obs_vector

        # 7. Info 딕셔너리에 디버깅 정보 추가
        info['reward_breakdown'] = {
            'xy_error': -self.w_xy * (1 - np.exp(-0.5 * xy_error**2)),
            'z_error': -self.w_z * (1 - np.exp(-0.5 * z_error**2)),
            'orientation_error': -0.1 * orientation_error,
            'success_bonus': self.success_reward if is_success else 0.0
        }

        # 8. 최종 계산된 값들을 반환합니다.
        return obs, float(reward), terminated, truncated, info


    
#---------------------------texture wrapper---------------
class TextureWrapper(gym.Wrapper):
    def __init__(self, env, texture_path):
        super().__init__(env)
        self.texture_path = os.path.abspath(texture_path)
        self.texture_id = -1
        self._client = getattr(self.env.unwrapped, "CLIENT", 0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_texture()
        return obs, info

    def _apply_texture(self):
        try:
            if self.texture_id == -1 and os.path.exists(self.texture_path):
                self.texture_id = p.loadTexture(self.texture_path, physicsClientId=self._client)
            if self.texture_id != -1:
                p.changeVisualShape(
                    self.env.unwrapped.PLANE_ID,
                    -1,
                    textureUniqueId=self.texture_id,
                    physicsClientId=self._client
                )
        except Exception as e:
            print(f"[TextureWrapper] 텍스처 적용 실패: {e}")




# -------------------- 메인 실행 코드 --------------------
if __name__ == "__main__":
    
    # 1) 기본 환경 생성
    INIT_XYZS = np.array([[0, 0, 10.0]])
    base_env = CustomHoverAviary(gui=True,
                                initial_xyzs=INIT_XYZS,
                                act=ActionType.VEL)
    base_env = TimeLimit(base_env, max_episode_steps=2400)

    # 2) MovingTargetWrapper로 타겟 추가
    env = MovingTargetWrapper(base_env, camera_size=(256, 256), obs_mode="rel_pos")

    # 3) TextureWrapper로 바닥 텍스처 적용
    texture_file = "gym_pybullet_drones/examples/textures/floor1.jpg"  # 원하는 텍스처 파일
    env = TextureWrapper(env, texture_file)

    # 4) BoundaryWrapper로 경계 패널티 추가
    env = BoundaryWrapper(env, x_max=25.0, y_max=25.0, z_min=1.0, z_max=20.0)
    # 디버그용 카메라의 위치와 각도를 설정합니다.
    p.resetDebugVisualizerCamera(
        cameraDistance=20,      # 카메라와 타겟 사이의 거리 (값을 키울수록 멀어짐)
        cameraYaw=45,           # 카메라의 수평 회전 각도 (정면 = 0)
        cameraPitch=-30,        # 카메라의 수직 기울기 (위에서 아래로 보는 각도)
        cameraTargetPosition=[0, 0, 0] # 카메라가 바라보는 지점 (월드 좌표)
    )
    # --- 테스트 루프 ---
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    t0 = time.time()
    ep_len = 0
    while time.time() - t0 < 60: # 60초 동안 테스트
        a = env.action_space.sample() # 랜덤 행동
        obs, r, term, trunc, info = env.step(a)
        ep_len += 1

        # 화면에 이미지 띄우기 (OpenCV 필요: pip install opencv-python)
        if isinstance(obs, dict) and "rgb" in obs:
            # both 모드
            frame = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            # rgb 모드
            frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        elif isinstance(obs, np.ndarray) and obs.ndim == 1:
            # rel_pos 모드 → 좌표값 출력
            #print("Relative position:", obs)
            frame = None
        else:
            frame = None

        if frame is not None:
            cv2.imshow('Drone Camera', cv2.resize(frame, (512, 512)))
            cv2.moveWindow('Drone Camera', 0, 0)
            
        if term or trunc:
            print(f"Episode ended after {ep_len} steps. Resetting...")
            obs, info = env.reset()
            ep_len = 0
        
        time.sleep(10./240.)
            
    env.close()
    print("done.")