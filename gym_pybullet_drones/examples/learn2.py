import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# 커스텀 환경 불러오기
from moving_car_test import make_custom_env, MovingTargetWrapper

#observation mode 설정
obs_mode = "rel_pos"

if __name__ == "__main__":
    # ---------------- CUDA 선택 ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")

    # ---------------- 환경 생성 ----------------
    def make_env():
        return make_custom_env(gui=False, obs_mode=obs_mode)

    train_env = DummyVecEnv([make_env])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecMonitor(eval_env)


    # ---------------------------
    # Policy 자동 선택
    # ---------------------------
    if obs_mode == "rel_pos":
        policy = "MlpPolicy"        # 단순 좌표 → MLP
    elif obs_mode == "rgb":
        policy = "CnnPolicy"        # 이미지 → CNN
    elif obs_mode == "both":
        policy = "MultiInputPolicy" # Dict(rgb+좌표) → MultiInput
    else:
        raise ValueError(f"Unknown obs_mode {obs_mode}")
    # ---------------- PPO 모델 ----------------
    # MultiInputPolicy → Dict(obs) 지원 (rgb + rel_pos)
    model = PPO(
        policy,
        env=train_env,
        verbose=1,
        tensorboard_log="./ppo_drone_tensorboard_multi/",       #terminal: tensorboard --logdir ./ppo_drone_tensorboard_multi/
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device,   # ← CUDA 사용 여부 자동 반영
    )

    # ---------------- 콜백 ----------------
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=20000, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_multi/",
        log_path="./logs_multi/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )

    # ---------------- 학습 실행 ----------------
    model.learn(total_timesteps=500_000, callback=eval_callback)

    # ---------------- 모델 저장 ----------------
    model.save("ppo_drone_multi_final")

    # ---------------- 평가 ----------------
    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        # rgb 모드에서는 render 필요 X (PyBullet GUI 꺼져있음)
        # print(reward)
