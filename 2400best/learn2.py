import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback, CheckpointCallback
from collections import Counter


# 커스텀 환경 불러오기
from moving_car_test import make_custom_env, MovingTargetWrapper

#observation mode 설정
obs_mode = "rel_pos"
# CPU 코어 수 기반으로 env 개수 설정 (최대 8 정도까지만)
MAX_ENVS = 8
cpu_count = os.cpu_count() or 4
num_envs = min(cpu_count, MAX_ENVS)
print(f"Using {num_envs} parallel envs with SubprocVecEnv")

class PeriodicVecNormalizeSave(BaseCallback):
    """
    VecNormalize 통계(.pkl)를 주기적으로 저장하는 콜백.
    EvalCallback의 최고 점수 갱신 여부와 "상관없이" 저장합니다.
    """
    def __init__(self, save_path: str, freq: int, verbose=1):
        super(PeriodicVecNormalizeSave, self).__init__(verbose)
        self.save_path = save_path
        self.freq = freq

    def _on_step(self) -> bool:
        # self.n_calls는 BaseCallback에 의해 1씩 증가하는 총 스텝 수입니다.
        if self.n_calls % self.freq == 0:
            if self.verbose > 0:
                print(f"Saving VecNormalize statistics to {self.save_path} (step {self.n_calls})")
            # self.training_env는 model.learn()에 전달된 train_env입니다.
            self.training_env.save(self.save_path)
        return True
    
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
    
if __name__ == "__main__":
    # ---------------- CUDA 선택 ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        num_envs = 8  # CUDA 사용 시 병렬 환경 수 증가
    #device = "cpu"
    print(f"Using device: {device}")

    # ---------------- 환경 생성 ----------------
    def make_env(rank: int, seed: int = 0):
        """
        rank: 몇 번째 env인지 (0 ~ num_envs-1)
        """
        def _init():
            env = make_custom_env(gui=False, obs_mode=obs_mode, is_test_mode=False)
            # 시드 다르게 주고 싶으면
            env.reset(seed=seed + rank)
            return env
        return _init
    # train은 SubprocVecEnv로 병렬
    train_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # eval은 속도 중요하지 않으니 DummyVecEnv 그대로 써도 되고, Subproc로 맞춰도 됨
    def make_eval_env():
        return make_custom_env(gui=False, obs_mode=obs_mode, is_test_mode=False)

    
    eval_env = DummyVecEnv([make_eval_env for _ in range(num_envs)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms


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
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device,   # ← CUDA 사용 여부 자동 반영
    )

    # ---------------- 콜백 ----------------
    save_folder = "./models_multi/"
    log_folder = "./logs_multi/"
    # "최신" 통계 파일 경로 (주기적으로 덮어쓰기됨)
    latest_vecnorm_path = os.path.join(save_folder, "latest_vecnormalize.pkl")
    # "최종" 모델/통계 파일 경로
    final_model_path = os.path.join(save_folder, "ppo_drone_multi_final")
    final_vecnorm_path = os.path.join(save_folder, "final_vecnormalize.pkl")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_folder,
        log_path=log_folder,
        eval_freq=5000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        callback_on_new_best=None
    )
    checkpoint_callback = CheckpointCallback(
      save_freq=100000,
      save_path=save_folder,
      name_prefix="ppo_backup", # (결과: ppo_backup_10000_steps.zip)
      save_vecnormalize=True   # (중요!) VecNormalize 사용 유무
    )
    # 3. (추가) "최신" 통계를 스텝마다 저장하는 콜백 생성(CheckpointCallback가 대체하므로 주석처리)
    #periodic_save_cb = PeriodicVecNormalizeSave(save_path=latest_vecnorm_path, freq=100000)
    # 4. (수정) callback 리스트에 두 콜백을 모두 전달
    termination_stats_cb = TerminationStatsCallback()

    callback_list = [eval_callback, checkpoint_callback, termination_stats_cb]
    model.learn(total_timesteps=100_000_000, callback=callback_list)

    # ---------------- 모델 저장 ----------------
    model.save(final_model_path)
    train_env.save(final_vecnorm_path)

    # ---------------- 평가 ----------------
    obs = eval_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        # rgb 모드에서는 render 필요 X (PyBullet GUI 꺼져있음)
        # print(reward)
