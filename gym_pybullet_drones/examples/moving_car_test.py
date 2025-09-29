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
    env = BoundaryWrapper(env, x_max=25.0, y_max=25.0, z_min=1.0, z_max=20.0)
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
                 init_target_z=0.5,
                 target_urdf="cube.urdf", target_rgba=(1.0, 0.2, 0.2, 1.0),
                 x_max=25.0, y_max=25.0,  # 타겟 이동 가능 범위(수평)
                 speed=0.01,

                 # ---- 추가: 보상/종료 파라미터 ----
                 desired_alt_above=10.0,  # 타겟 '상공' 목표 고도 차
                 xy_tol=1.5,              # 성공 밴드(수평)
                 z_tol=1.0,               # 성공 밴드(수직)
                 max_xy=40.0,             # 너무 멀어지면 종료
                 min_z=0.2,               # 거의 충돌 수준
                 alive_bonus=0.05,        # 매 스텝 생존 보너스
                 w_xy=1.0,                # 수평 오차 가중치
                 w_z=0.7,                 # 수직 오차 가중치
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
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            )
        elif self.obs_mode == "both":
            self.observation_space = spaces.Dict({
                "rgb": spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8),
                "rel_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            })
        self.action_space = self.env.action_space


        # 움직임 관련
        self.x_max, self.y_max = x_max, y_max
        self.speed = speed
        self.target_pos = np.array([0.0, 0.0, self.init_target_z])
        self.target_vel = np.random.uniform(-self.speed, self.speed, size=2)

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
                #globalScaling=20.0,
            )

            # 텍스처 적용 코드 추가
            texture_path = "gym_pybullet_drones/examples/textures/moving_obj.jpg"
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
        """경계 내에서 타겟 위치 업데이트"""
        self.target_pos[0] += self.target_vel[0]
        self.target_pos[1] += self.target_vel[1]

        # 경계 충돌 처리 (반사)
        if abs(self.target_pos[0]) > self.x_max:
            self.target_vel[0] *= -1
            self.target_pos[0] = np.clip(self.target_pos[0], -self.x_max, self.x_max)
        if abs(self.target_pos[1]) > self.y_max:
            self.target_vel[1] *= -1
            self.target_pos[1] = np.clip(self.target_pos[1], -self.y_max, self.y_max)

        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        p.resetBasePositionAndOrientation(
            self._target_id, self.target_pos, orn, physicsClientId=self._client
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

        # obs 모드별 반환
        if self.obs_mode == "rgb":
            rgb = self._render_rgb()
            obs = rgb
        elif self.obs_mode == "rel_pos":
            obs = rel_pos.astype(np.float32)
        elif self.obs_mode == "both":
            rgb = self._render_rgb()
            obs = {"rgb": rgb, "rel_pos": rel_pos.astype(np.float32)}

        return obs, info


    def step(self, action):
        obs_base, reward_base, term_base, trunc_base, info = self.env.step(action)

        # 타겟 업데이트
        self._update_target()

        # 현재 상대좌표
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(self._target_id, physicsClientId=self._client)
        rel_pos = np.array(target_pos) - np.array(drone_pos)   # [dx, dy, dz]
        dx, dy, dz = rel_pos
        dist_xy = float(np.hypot(dx, dy))
        z_err   = float(dz + self.desired_alt_above)  # 목표상태에서: dz = -desired_alt_above → z_err=0

        # 보상 구성
        reward = 0.0
        reward += self.alive_bonus
        reward -= self.w_xy * (dist_xy**2)
        reward -= self.w_z  * (z_err**2)

        # 액션 패널티(연속액션 가정)
        try:
            a_norm = float(np.linalg.norm(np.asarray(action)))
            reward -= self.w_act * (a_norm**2)
        except Exception:
            pass

        # 드론 속도 패널티(선택)
        try:
            lin_vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=self._client)
            v = float(np.linalg.norm(lin_vel))
            reward -= self.w_vel * (v**2)
        except Exception:
            pass

        # 성공 밴드 추가 보상 (수평/수직 모두 허용 오차 내)
        if (dist_xy <= self.xy_tol) and (abs(z_err) <= self.z_tol):
            reward += self.success_reward

        # 종료/트렁케이션 조건
        # 1) 너무 낮음(충돌 근접)
        term_local = (drone_pos[2] <= self.min_z)

        # 2) 너무 멀리 이탈한 상태가 오래 지속
        if dist_xy > self.max_xy:
            self._far_counter += 1
        else:
            self._far_counter = 0
        term_local = term_local or (self._far_counter >= self.patience_steps)

        # TimeLimit은 외부 TimeLimit wrapper가 처리 → 여기서는 truncated_local False 유지
        # 종료/트렁케이션 플래그 합치기
        terminated = bool(term_base or term_local)
        truncated  = bool(trunc_base)   # TimeLimit은 그대로 유지

        # obs 모드별 반환
        
        if self.obs_mode == "rgb":
            rgb = self._render_rgb()
            obs = rgb
        elif self.obs_mode == "rel_pos":
            obs = rel_pos.astype(np.float32)
        elif self.obs_mode == "both":
            rgb = self._render_rgb()
            obs = {"rgb": rgb, "rel_pos": rel_pos.astype(np.float32)}

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
            
    env.close()
    print("done.")