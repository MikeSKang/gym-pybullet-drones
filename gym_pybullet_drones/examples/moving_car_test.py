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

        # 하드 종료는 없애고, TimeLimit만 사용
        return obs, reward, False, False, info

# -------------------- MovingTargetWrapper --------------------
class MovingTargetWrapper(gym.Wrapper):
    """
    - 원본 환경 위에 타겟 오브젝트(큐브)를 추가하고,
    - 사각형 범위 내에서 자동으로 움직이도록 합니다.
    - 드론 시점의 카메라 RGB 이미지를 관측으로 반환합니다.
    """
    def __init__(
        self,
        env,
        camera_size=(84, 84),
        fov_deg=90.0,
        near=0.02,
        far=20.0,
        init_target_z=0.5,
        target_urdf="cube.urdf",
        target_rgba=(1.0, 0.2, 0.2, 1.0),
        x_max=25.0,
        y_max=25.0,
        speed=0.01,
    ):
        super().__init__(env)

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
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8
        )
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
        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self._spawn_or_reset_target([0.0, 0.0, self.init_target_z], orn)

        # 랜덤 속도 초기화
        self.target_vel = np.random.uniform(-self.speed, self.speed, size=2)
        self.target_pos = np.array([0.0, 0.0, self.init_target_z])

        rgb = self._render_rgb()
        return rgb, info

    def step(self, action):
        obs_base, reward_base, terminated, truncated, info = self.env.step(action)

        # 타겟 업데이트
        self._update_target()

        rgb = self._render_rgb()
        reward = float(reward_base)
        return rgb, reward, terminated, truncated, info

    
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
    env = MovingTargetWrapper(base_env, camera_size=(128, 128))

    # 3) TextureWrapper로 바닥 텍스처 적용
    texture_file = "gym_pybullet_drones/examples/textures/floor1.jpg"  # 원하는 텍스처 파일
    env = TextureWrapper(env, texture_file)

    # 4) BoundaryWrapper로 경계 패널티 추가
    env = BoundaryWrapper(env, x_max=25.0, y_max=25.0, z_min=1.0, z_max=20.0)

    # --- 테스트 루프 ---
    obs, info = env.reset()
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    
    t0 = time.time()
    ep_len = 0
    while time.time() - t0 < 60: # 60초 동안 테스트
        a = env.action_space.sample() # 랜덤 행동
        obs, r, term, trunc, info = env.step(a)
        ep_len += 1

        # 화면에 이미지 띄우기 (OpenCV 필요: pip install opencv-python)
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        cv2.imshow('Drone Camera', cv2.resize(frame, (512, 512)))
        cv2.moveWindow('Drone Camera', 0, 0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if term or trunc:
            print(f"Episode ended after {ep_len} steps. Resetting...")
            obs, info = env.reset()
            ep_len = 0
            
    env.close()
    print("done.")