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
from gym_pybullet_drones.utils.utils import sync


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # PPO가 내부 env에서 접근할 때를 대비해 기본값 생성
        self.prev_action = np.zeros(4, dtype=np.float32)
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
    def __init__(self,
                 env,
                 obs_mode="rel_pos",
                 camera_size=(84, 84),
                 init_target_z=0.03,
                 x_max=25.0, # 자동차 활동 반경
                 y_max=25.0,
                 ):
        super().__init__(env)
        self.obs_mode = obs_mode
        
        # =====================================================================
        # 1. 보상 및 종료 하이퍼파라미터 정의
        # =====================================================================
        # 이 값들을 조정하여 에이전트의 학습 방향을 튜닝할 수 있습니다.
        self.w_dist     = 1.0      # (핵심) 거리 보상 가중치
        self.w_approach = 0.2      # 접근 속도 보상 가중치
        self.w_speed    = 0.05     # 드론 속도 페널티 가중치
        self.w_alt      = 0.1      # 고도 페널티 가중치
        
        self.desired_dist    = 10.0     # 목표 유지 거리 [m]
        self.dist_sharpness  = 0.1      # 거리 보상 곡선의 뾰족함 정도 (값이 클수록 좁고 뾰족해짐)
        
        self.max_drone_speed = 10.0     # 페널티가 시작되는 드론 속도 [m/s]
        self.alt_range       = [5.0, 20.0] # 이상적인 고도 범위 [m]

        self.min_dist_fail   = 2.0      # 실패(너무 가까움) 기준 [m]
        self.max_dist_fail   = 30.0     # 실패(너무 멂) 기준 [m]
        self.min_z_crash     = 0.2      # 실패(추락) 기준 [m]
        
        self.dt = self.env.unwrapped.CTRL_TIMESTEP

        # =====================================================================
        # 2. PyBullet 및 환경 기본 설정
        # =====================================================================
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.init_target_z = float(init_target_z)
        self._target_id = None
        self._target_urdf = "cube_small.urdf"
        self._client = getattr(self.env.unwrapped, "CLIENT", 0)
        self._drone_id = getattr(self.env.unwrapped, "DRONE_IDS", [0])[0]

        # =====================================================================
        # 3. 카메라 및 관측/행동 공간 정의
        # =====================================================================
        self.W, self.H = int(camera_size[0]), int(camera_size[1])
        self._proj = p.computeProjectionMatrixFOV(fov=90.0, aspect=self.W/self.H, nearVal=0.02, farVal=20.0)
        
        if self.obs_mode == "rgb":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        elif self.obs_mode == "rel_pos":
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        elif self.obs_mode == "both":
            self.observation_space = spaces.Dict({
                "rgb": spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8),
                "rel_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
            })
        self.action_space = self.env.action_space

        # =====================================================================
        # 4. 자동차(타겟) 움직임 관련 변수 초기화
        # =====================================================================
        self.x_max, self.y_max = x_max, y_max
        self.target_pos = np.array([0.0, 0.0, self.init_target_z])
        
        self.max_speed = 2.0
        self.max_accel = 0.05
        self.max_turn_rate = 0.05
        
        self.speed = 0.05
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.turn_rate = 0.0
        self.accel = 0.0

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
                globalScaling=1.0,
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
        
        #타겟 위치 랜덤성
        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        tx = np.random.uniform(-5.0, 5.0)
        ty = np.random.uniform(-5.0, 5.0)
        self._spawn_or_reset_target([tx, ty, self.init_target_z], orn)

        # 드론 시작 위치도 랜덤하게 (타겟과 일정 거리 유지)
        rand_r = np.random.uniform(6.0, 18.0)
        rand_th = np.random.uniform(0, 2*np.pi)
        dx, dy = rand_r * np.cos(rand_th), rand_r * np.sin(rand_th)
        drone_pos0 = np.array([dx, dy, np.random.uniform(8.0, 14.0)])
        p.resetBasePositionAndOrientation(
            self._drone_id,
            drone_pos0.tolist(),
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._client
        )

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
        self.prev_action = np.zeros(4, dtype=np.float32)
        full_obs_vector = np.concatenate([rel_pos, lin_vel, ang_vel, self.prev_action.flatten()]).astype(np.float32)
        
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
        # 1. 하위 환경 실행 및 타겟 업데이트
        obs_base, _, term_base, trunc_base, info = self.env.step(action)
        self._update_target()
        
        

        # 2. 드론 및 타겟 상태 정보 수집
        drone_pos, _ = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(self._target_id, physicsClientId=self._client)
        rel_pos = np.array(target_pos) - np.array(drone_pos)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=self._client)

        dist_vec = np.array(target_pos) - np.array(drone_pos)
        dist_3d = np.linalg.norm(dist_vec)
        if not hasattr(self, "_prev_dist"):
            self._prev_dist = dist_3d

        # 3. 보상(Reward) 계산
        reward = 0.0
        reward += 0.005  # Alive Bonus

        # (A) 거리 보상 (종 모양 곡선): 목표 거리에 가까울수록 보상이 커짐
        error = dist_3d - self.desired_dist
        dist_reward = np.exp(-self.dist_sharpness * (error**2))
        reward += self.w_dist * dist_reward
        
        # (B) 접근 속도 보상: 타겟 방향으로 움직이면 보너스
        dist_xy = np.linalg.norm(dist_vec[:2])
        if dist_xy > 1e-6:
            approach_speed = np.dot(lin_vel[:2], dist_vec[:2] / dist_xy)
            reward += self.w_approach * approach_speed

        progress = self._prev_dist - dist_3d
        reward += 0.5 * progress   # 가중치(0.3~1.0)로 조절
        self._prev_dist = dist_3d
            
        # (C) 속도 페널티: 너무 빠르게 움직이면 페널티
        speed = np.linalg.norm(lin_vel)
        speed_margin = max(0, speed - self.max_drone_speed)
        reward -= self.w_speed * speed_margin
        
        # (D) 고도 페널티: 이상적인 고도 범위를 벗어나면 페널티
        alt_margin = max(0, self.alt_range[0] - drone_pos[2]) + max(0, drone_pos[2] - self.alt_range[1])
        reward -= self.w_alt * alt_margin

        # (E) 행동 페널티: 너무 큰 행동을 취하면 페널티
        action_penalty = np.linalg.norm(action) # 행동 벡터의 크기
        reward -= 0.003 * action_penalty # 작은 가중치를 곱해 페널티로 사용

        # 4. 종료(Termination) 조건 계산
        terminated = False
        fail_far = (dist_3d > self.max_dist_fail)
        fail_close = (dist_3d < self.min_dist_fail)
        fail_crash = (drone_pos[2] <= self.min_z_crash)
        
        # 실패 조건 중 하나라도 만족하면 에피소드 종료 및 페널티 부과
        if fail_far or fail_close or fail_crash:
            reward -= 10.0
            terminated = True
        
        # 하위 환경의 종료 신호(term_base)를 반영
        if not terminated:
            terminated = bool(term_base)
        # 하위 환경의 시간 초과 신호(trunc_base)를 반영
        truncated = bool(trunc_base)

        # 5. 관측(Observation) 정보 구성 및 반환
        rel_pos = dist_vec
        full_obs_vector = np.concatenate([rel_pos, lin_vel, ang_vel, self.prev_action.flatten()]).astype(np.float32)
        self.prev_action = np.array(action).flatten()
        if self.obs_mode == "rgb":
            obs = self._render_rgb()
        elif self.obs_mode == "rel_pos":
            obs = full_obs_vector
        elif self.obs_mode == "both":
            obs = {"rgb": self._render_rgb(), "rel_pos": full_obs_vector}
        else:
            obs = full_obs_vector
            
        return obs, float(reward), bool(terminated), bool(truncated), info


    
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
    TIMESTEP = env.CTRL_TIMESTEP
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    t0 = time.time()
    ep_len = 0
    start_time = time.time()

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
        
        sync(ep_len, start_time, TIMESTEP)
            
    env.close()
    print("done.")