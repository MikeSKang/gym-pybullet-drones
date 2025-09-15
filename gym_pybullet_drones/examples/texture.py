import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import numpy as np
import time
import os
import gym_pybullet_drones


class TextureWrapper(gym.Wrapper):
    def __init__(self, env, texture_path):
        super().__init__(env)
        self.texture_path = os.path.abspath(texture_path)
        self.texture_id = -1
        self._fix_observation_space_dtype()

    # 관측공간 dtype을 float32로 통일
    def _fix_observation_space_dtype(self):
        space = self.env.observation_space
        if isinstance(space, spaces.Box):
            low = np.array(space.low, dtype=np.float32)
            high = np.array(space.high, dtype=np.float32)
            self.observation_space = spaces.Box(
                low=low, high=high, shape=space.shape, dtype=np.float32
            )
        elif isinstance(space, spaces.Dict):
            self.observation_space = spaces.Dict({
                k: spaces.Box(
                    low=np.array(s.low, dtype=np.float32),
                    high=np.array(s.high, dtype=np.float32),
                    shape=s.shape, dtype=np.float32
                ) if isinstance(s, spaces.Box) else s
                for k, s in space.spaces.items()
            })
        else:
            self.observation_space = space  # 다른 타입은 그대로

    def _process_obs(self, obs):
        # numpy array이면 float32로 변환 후 clip
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32, copy=False)
            if isinstance(self.observation_space, spaces.Box):
                obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        # dict면 key별로 처리
        elif isinstance(obs, dict):
            obs = {k: (np.clip(v.astype(np.float32, copy=False),
                               self.observation_space[k].low,
                               self.observation_space[k].high)
                       if isinstance(v, np.ndarray) and isinstance(self.observation_space[k], spaces.Box)
                       else v)
                   for k, v in obs.items()}
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._apply_texture()
        obs = self._process_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, reward, terminated, truncated, info

    def _apply_texture(self):
        try:
            if self.texture_id == -1:
                self.texture_id = p.loadTexture(
                    self.texture_path, physicsClientId=self.env.unwrapped.CLIENT
                )
            if self.texture_id != -1:
                p.changeVisualShape(
                    self.env.unwrapped.PLANE_ID, -1,
                    textureUniqueId=self.texture_id,
                    physicsClientId=self.env.unwrapped.CLIENT
                )
        except Exception as e:
            print(f"[TextureWrapper] 텍스처 적용 실패: {e}")


# --- 메인 ---
env = gym.make("hover-aviary-v0", gui=True)

texture_file = os.path.join(
    os.path.dirname(gym_pybullet_drones.__file__),
    "examples", "grass_texture.png"
)

env = TextureWrapper(env, texture_path=texture_file)

obs, info = env.reset()
start = time.time()
while time.time() - start < 10:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    #if terminated or truncated:
    #    obs, info = env.reset()
env.close()
print("시뮬레이션 완료!")
