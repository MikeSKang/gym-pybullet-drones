import pickle
# from stable_baselines3.common.vec_env import VecNormalize  # 이 임포트는 필요 없습니다.

# pkl 파일에서 VecNormalize 객체 로드
with open("./models_multi/vecnormalize.pkl", "rb") as f:
    vec_normalize_stats = pickle.load(f)

# 불러온 객체의 타입 확인 (VecNormalize 클래스 인스턴스임을 알 수 있습니다)
print(f"객체 타입: {type(vec_normalize_stats)}")


# === 통계 정보 확인 (속성 접근) ===

print("--- 관측 정규화 통계 ---")
# 'obs_rms'와 'ret_rms'는 RMS(Root Mean Square) 객체이며, 다시 그 객체의 'mean' 속성에 접근해야 합니다.
print(f"평균(mean): {vec_normalize_stats.obs_rms.mean}")
print(f"분산(var): {vec_normalize_stats.obs_rms.var}")
# print(f"카운트(count): {vec_normalize_stats.obs_rms.count}") # 통계 업데이트 횟수

print("\n--- 보상 정규화 통계 ---")
print(f"평균(mean): {vec_normalize_stats.ret_rms.mean}")
print(f"분산(var): {vec_normalize_stats.ret_rms.var}")
