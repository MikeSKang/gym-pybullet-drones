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
def make_custom_env(gui=False, obs_mode="rel_pos", is_test_mode=False):
    INIT_XYZS = np.array([[0, 0, 2.0]])
    base_env = CustomHoverAviary(gui=gui, initial_xyzs=INIT_XYZS, act=ActionType.VEL)
    base_env = TimeLimit(base_env, max_episode_steps=900)

    env = MovingTargetWrapper(base_env, camera_size=(84, 84), obs_mode=obs_mode, is_test_mode=is_test_mode)
    env = TextureWrapper(env, "gym_pybullet_drones/examples/textures/floor1.jpg")
    #env = BoundaryWrapper(env, x_max=25.0, y_max=25.0, z_min=1.0, z_max=20.0)
    return env

# -------------------- Custom HoverAviary --------------------
class CustomHoverAviary(HoverAviary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, ctrl_freq=30)
        
        # PPO가 내부 env에서 접근할 때를 대비해 기본값 생성
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.SPEED_LIMIT = 10.0
    def _computeTerminated(self):
        return False  # 성공 조건 무시 (학습 목적이면 직접 정의해도 됨)

    def _computeTruncated(self):
        return False  # 활동 반경 제한 및 시간 제한 무시

# -------------------- BoundaryWrapper --------------------
class BoundaryWrapper(gym.Wrapper):
    def __init__(self, env, x_max=5.0, y_max=5.0, z_min=1.0, z_max=15.0):
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
                 init_target_z=0.2,
                 x_max=5.0, # 자동차 활동 반경
                 y_max=5.0,
                 is_test_mode=False
                 ):
        super().__init__(env)
        self.obs_mode = obs_mode
        self.is_test_mode = is_test_mode
        
        # =====================================================================
        # 1. 보상 및 종료 하이퍼파라미터 정의
        # =====================================================================
        # 1. 목표 (보상) - "타겟에 붙어라"
        self.w_dist = 10.0       # (수정) 기본 가중치 10.0 (가장 중요)
        self.desired_dist = 0.0     
        self.dist_sharpness = 1.5     
        
        # 2. 비용 (페널티) - "물리적으로 흔들리지 마라"
        self.w_ang_vel = 0.01     # (수정) 0.2 -> 0.01 (페널티 스케일 조정)

        # 3. 비용 (페널티) - "명령을 급격히 바꾸지 마라"
        self.w_action_rate = 0.01  # (수정) 0.1 -> 0.01 (페널티 스케일 조정)
        # 4. 비용 (페널티) - "게으른 상승을 하지 마라"
        #self.w_alt_penalty = 0.1
        #self.ideal_alt = 2.0 # (reset 시 드론의 기본 고도)
        #바닥접근 패널티
        self.w_crash_avoid = 0.5

        self.max_drone_speed = 10.0    

        self.min_dist_fail = 0.5      
        self.CAMERA_FOV_THRESHOLD = 45.0 
        self.max_z_fail = 5.0
        self.min_z_crash = 0.2    
        self.lost_steps = 0  # 타겟 놓친 시간 카운터
        
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
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        elif self.obs_mode == "both":
            self.observation_space = spaces.Dict({
                "rgb": spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8),
                "rel_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
            })
        self.action_space = self.env.action_space

        # =====================================================================
        # 4. 자동차(타겟) 움직임 관련 변수 초기화
        # =====================================================================
        self.x_max, self.y_max = x_max, y_max
        self.target_pos = np.array([0.0, 0.0, self.init_target_z])
        
        self.min_speed = 0.1
        self.max_speed = 0.8
        self.max_accel = 0.05
        self.max_turn_rate = 0.05
        
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.turn_rate = np.random.uniform(-self.max_turn_rate, self.max_turn_rate)
        self.accel = np.random.uniform(-self.max_accel, self.max_accel)

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
                globalScaling=6.0,
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
        self.accel += np.random.uniform(-0.01, 0.01)
        self.turn_rate += np.random.uniform(-0.05, 0.05)
        
        # 2. 가속도와 회전율을 최대값으로 제한합니다.
        self.accel = np.clip(self.accel, -self.max_accel, self.max_accel)
        self.turn_rate = np.clip(self.turn_rate, -self.max_turn_rate, self.max_turn_rate)

        # 3. 현재 속도와 진행 방향 각도를 업데이트합니다.
        self.speed += self.accel
        self.speed = np.clip(self.speed, self.min_speed, self.max_speed) # 속도는 0 이상, 최대 속도 이하
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
            # [수정] self._img_flags 대신 상수를 직접 사용
            flags=p.ER_NO_SEGMENTATION_MASK, 
            physicsClientId=self._client,
        )
        rgb = np.reshape(rgb, (self.H, self.W, 4))[:, :, :3]
        return rgb.astype(np.uint8, copy=False)

    # -------------------- Gym API --------------------
    def reset(self, **kwargs):
        self.lost_steps = 0
        obs_base, info = self.env.reset(**kwargs)
        # self.env.reset()으로 시뮬레이션이 초기화되었으므로,
        # 이전 타겟 ID는 더 이상 유효하지 않습니다. None으로 초기화합니다.
        self._target_id = None
        
        # 1. 드론 위치 고정 (0, 0, 2.0)
        drone_pos = np.array([0.0, 0.0, 2.0])
        p.resetBasePositionAndOrientation(
            self._drone_id,
            drone_pos.tolist(),
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self._client
        )

        # 2. 타겟을 "시야각 원뿔" 내부에 스폰
        
        # 2a. 드론과 타겟 평면의 수직 거리 계산
        dz = drone_pos[2] - self.init_target_z # (2.0 - 0.03 = 1.97m)
        
        # 2b. CAMERA_FOV_THRESHOLD 시야각이 바닥에서 만드는 최대 수평 반경(radius) 계산
        # tan(theta) = radius / dz
        max_radius = dz * np.tan(np.radians(self.CAMERA_FOV_THRESHOLD-30.0)) # (1.97 * tan(CAMERA_FOV_THRESHOLD - 10.0)) #안정성 위해 10도 더 좁게

        # 2c. 이 반경(max_radius) 안에서 랜덤한 (x, y) 좌표 생성
        rand_r = np.random.uniform(0.0, max_radius)
        rand_th = np.random.uniform(0, 2 * np.pi)
        
        tx = rand_r * np.cos(rand_th)
        ty = rand_r * np.sin(rand_th)
        target_pos = np.array([tx, ty, self.init_target_z])
        
        # 2d. 타겟 스폰
        orn = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        self._spawn_or_reset_target(target_pos.tolist(), orn)

        # 랜덤 속도 초기화
        #self.target_vel = np.random.uniform(-self.speed, self.speed, size=2)
        self.target_pos = target_pos

        # 좌표 계산
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(self._target_id, physicsClientId=self._client)
        rel_pos = np.array(target_pos) - np.array(drone_pos)

        # 자기인식 정보 추가: 드론의 속도와 각속도
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=self._client)
        # 타겟 속도 (self.speed, self.angle은 __init__ / _update_target에서 관리 중)
        vx = self.speed * np.cos(self.angle)
        vy = self.speed * np.sin(self.angle)
        target_vel = np.array([vx, vy, 0.0], dtype=np.float32)
        # 상대 속도 = 타겟 - 드론
        rel_vel = target_vel - np.array(lin_vel, dtype=np.float32)

        #드론의 RPY 각도 계산
        rpy = p.getEulerFromQuaternion(drone_orn)
        self.prev_action = np.zeros(4, dtype=np.float32)
        full_obs_vector = np.concatenate([rel_pos, rel_vel, ang_vel, rpy, self.prev_action.flatten()]).astype(np.float32)
        
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
        drone_pos, drone_orn = p.getBasePositionAndOrientation(self._drone_id, physicsClientId=self._client)
        target_pos, _ = p.getBasePositionAndOrientation(self._target_id, physicsClientId=self._client)
        rel_pos = np.array(target_pos) - np.array(drone_pos)
        lin_vel, ang_vel = p.getBaseVelocity(self._drone_id, physicsClientId=self._client)

        dist_vec = np.array(target_pos) - np.array(drone_pos)
        dist_3d = np.linalg.norm(dist_vec)
        dist_xy = np.linalg.norm(dist_vec[:2])

        # 3. 보상(Reward) 계산
        reward = 0.0   #alive bonus

        # (A) 거리 보상 (THE GOAL)
        error = dist_xy - self.desired_dist
        dist_reward = np.exp(-self.dist_sharpness * (error**2))
        reward += self.w_dist * dist_reward
        
        # --- (B) 안정성 페널티 (THE COST) ---
        
        # (B-1) 물리적 흔들림 (각속도) 페널티 (부드러운 제곱 사용)
        ang_vel_norm = np.linalg.norm(ang_vel)
        reward -= self.w_ang_vel * (ang_vel_norm**2) # 제곱 페널티

        # (B-2) 명령 흔들림 (행동 변화율) 페널티 (부드러운 제곱 사용)
        action_diff = action - self.prev_action
        action_rate_norm = np.linalg.norm(action_diff)
        reward -= self.w_action_rate * (action_rate_norm**2) # 제곱 페널티

        # (C) "바닥 접근" 페널티 (Lava Floor)
        # 0.2m(충돌)에 가까워질수록 페널티가 지수적으로 증가합니다.
        # (drone_pos[2]가 1.0m 근처면 페널티가 매우 작아짐)
        crash_proximity = drone_pos[2] - self.min_z_crash
        if crash_proximity < 1.0: # 1.2m 고도 아래로 내려오면 페널티 시작
            # np.exp(-5.0 * 0.0) == 1.0 (최대 페널티)
            # np.exp(-5.0 * 1.0) == 0.006 (거의 0)
            crash_prox_penalty = np.exp(-5.0 * crash_proximity)
            reward -= self.w_crash_avoid * crash_prox_penalty
        


        # 4. 종료(Termination) 조건 계산
        terminated = False
        term_reason = None # 종료 사유
        #각도 기반 실패 (test2.py의 로직을 가져옴)
        drone_down_vector = np.array([0.0, 0.0, -1.0])
        drone_to_target_vector = rel_pos # (rel_pos = dist_vec)
        
        reacq_angle_deg = 0.0 # (정확히 아래 있을 때)
        
        if dist_3d > 1e-6:
            dot_product = -drone_to_target_vector[2] # (0,0,-1) . (x,y,z) = -z
            cos_theta = np.clip(dot_product / dist_3d, -1.0, 1.0)
            reacq_angle_deg = np.degrees(np.arccos(cos_theta))

        fail_angle = (reacq_angle_deg > self.CAMERA_FOV_THRESHOLD)

        #고도 기반 실패
        fail_z_limit = (drone_pos[2] > self.max_z_fail)
        
        # (기존) 추락 및 근접 실패
        fail_close = (dist_3d < self.min_dist_fail)
        fail_crash = (drone_pos[2] <= self.min_z_crash)
        
        # 실패 조건 중 하나라도 만족하면 에피소드 종료 및 페널티 부과
        # hard 실패들
        hard_fail = fail_z_limit or fail_close or fail_crash

        if hard_fail:
            reward -= 10.0
            terminated = True
            if fail_crash:
                term_reason = "crash"
            elif fail_close:
                term_reason = "too_close"
            elif fail_z_limit:
                term_reason = "too_high"
        elif fail_angle:
            self.lost_steps += 1
            if self.is_test_mode:
                # [테스트 모드]: 신호를 보내고 종료하지 않음
                terminated = True
                info['status'] = 'TARGET_LOST'
                reward -= 10.0 # (타겟을 놓친 것에 대한 가벼운 페널티)
                term_reason = "angle_lost_test"
            else:
                reward -= 10.0
                terminated = True
                term_reason = "angle_lost_test"
        else:
            self.lost_steps = 0
        
        # 하위 환경의 종료 신호(term_base)를 반영
        if not terminated:
            terminated = bool(term_base)
            term_reason = term_reason or "base_terminated"
        # 하위 환경의 시간 초과 신호(trunc_base)를 반영
        truncated = bool(trunc_base)
        if truncated:
            term_reason = term_reason or "base_truncated"

        if terminated or truncated:
            info = dict(info)               # 혹시 래퍼에서 tuple 쓸 수 있으니 복사
            info["termination_reason"] = term_reason or "unknown"

        # 5. 관측(Observation) 정보 구성 및 반환
        rel_pos = dist_vec
        # 타겟 속도 (speed/angle은 바로 위에서 _update_target() 호출로 최신 값)
        vx = self.speed * np.cos(self.angle)
        vy = self.speed * np.sin(self.angle)
        target_vel = np.array([vx, vy, 0.0], dtype=np.float32)
        # 상대 속도 = 타겟 - 드론
        rel_vel = target_vel - np.array(lin_vel, dtype=np.float32)

        #드론의 RPY 각도 계산
        rpy = p.getEulerFromQuaternion(drone_orn)
        full_obs_vector = np.concatenate([rel_pos, rel_vel, ang_vel, rpy, self.prev_action.flatten()]).astype(np.float32)
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
    INIT_XYZS = np.array([[0, 0, 2.0]])
    base_env = CustomHoverAviary(gui=True,
                                initial_xyzs=INIT_XYZS,
                                act=ActionType.VEL)
    base_env = TimeLimit(base_env, max_episode_steps=300)

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