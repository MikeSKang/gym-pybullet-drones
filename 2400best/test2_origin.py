import os
import time
import pybullet as p

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from moving_car_test import make_custom_env
from gym_pybullet_drones.utils.utils import sync

# ---------------- 경로 설정 ----------------
MODEL_DIR = "./models_multi"
# learn2.py에서 저장한 것들:
# - best_model.zip
# - ppo_drone_multi_final.zip
# - latest_vecnormalize.pkl
# - final_vecnormalize.pkl
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
VECNORM_CANDIDATES = [
    os.path.join(MODEL_DIR, "final_vecnormalize.pkl"),
    os.path.join(MODEL_DIR, "latest_vecnormalize.pkl"),
]

OBS_MODE = "rel_pos"


def make_env_for_test():
    # 학습 때랑 같은 env를 만들되, 화면은 봐야 하니까 gui=True
    return make_custom_env(gui=True, obs_mode=OBS_MODE, is_test_mode=True)


if __name__ == "__main__":
    # 1) 학습 때처럼 DummyVecEnv로 한 겹 감싼다
    vec_env = DummyVecEnv([make_env_for_test])

    # 2) VecNormalize pkl을 로드해서 같은 정규화 통계를 쓰게 한다
    vecnorm_loaded = False
    for path in VECNORM_CANDIDATES:
        if os.path.exists(path):
            print(f"[info] loading VecNormalize stats from: {path}")
            vec_env = VecNormalize.load(path, vec_env)
            vecnorm_loaded = True
            break

    if not vecnorm_loaded:
        # pkl이 없으면 그냥 정규화 없이 진행
        print("[warn] VecNormalize .pkl not found, running without loaded stats")

    # 테스트이므로 정규화 업데이트는 멈춘다
    if isinstance(vec_env, VecNormalize):
        vec_env.training = False
        vec_env.norm_reward = False

    # 3) 모델을 로드할 때 env=vec_env 로 넣어준다
    model = PPO.load(MODEL_PATH, env=vec_env, device="auto")
    print(f"[ok] loaded model from {MODEL_PATH}")

    # 4) 안쪽 실제 pybullet env 꺼내기 (카메라/DRONE_ID용)
    # vec_env -> DummyVecEnv -> 실제 env
    base_env = vec_env.venv.envs[0]
    TIMESTEP = base_env.CTRL_TIMESTEP

    # 디버깅 카메라 초기 위치
    p.resetDebugVisualizerCamera(
        cameraDistance=5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
    )

    # 5) 테스트 루프
    # vec_env는 gymnasium 스타일이 아니라서 reset() 한 개만 받으면 돼
    obs = vec_env.reset()
    print("테스트 시작")

    for ep in range(5):
        done = False
        total_reward = 0.0
        steps = 0
        start = time.time()

        obs = vec_env.reset()

        # 실제 드론 id / client id 는 안쪽 env에서 가져와야 한다
        drone_id = base_env.DRONE_IDS[0]
        client_id = base_env.CLIENT

        while not done:
            # 정책으로부터 행동 추론
            action, _ = model.predict(obs, deterministic=True)

            # vec_env는 보통 4개를 돌려준다: obs, reward, done, info
            obs, reward, done, info = vec_env.step(action)

            # reward, done이 벡터일 수 있으니 첫 번째 것만
            if hasattr(reward, "__len__"):
                reward_val = float(reward[0])
            else:
                reward_val = float(reward)

            if hasattr(done, "__len__"):
                done_flag = bool(done[0])
            else:
                done_flag = bool(done)

            total_reward += reward_val
            steps += 1

            # 드론 위치를 따라가면서 카메라 이동
            drone_pos, _ = p.getBasePositionAndOrientation(
                drone_id, physicsClientId=client_id
            )
            p.resetDebugVisualizerCamera(
                cameraDistance=3,
                cameraYaw=0,
                cameraPitch=-70,
                cameraTargetPosition=drone_pos,
            )

            # 시뮬 타임 맞춰주기
            sync(steps, start, TIMESTEP)

            if done_flag:
                break

        print(f"[ep {ep+1}] reward={total_reward:.2f}, steps={steps}")

    vec_env.close()
    print("테스트 종료")
