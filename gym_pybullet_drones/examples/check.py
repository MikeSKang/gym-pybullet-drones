#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Siamese 가중치/좌표 투영 검증 스크립트 (보정 포함 완성본)

기능:
1) 센터 크롭 테스트: 서치(255x255)의 중앙 127x127을 템플릿으로 사용 → heatmap 중앙 확인
2) 랜덤 크롭 테스트: 가장자리 margin을 둔 랜덤 위치 템플릿 → 피크를 서치 좌표로 역투영 후 오차 통계
3) 좌표 역투영:
   - 기본: 센터 기준 + 칸폭 (선형 이론식)
   - 옵션: 축별 선형 보정(a,b) 자동 추정으로 실측 기반 보정
4) 옵션:
   --letterbox        : 종횡비 유지 + 패딩으로 서치 생성
   --imagenet-norm    : ImageNet mean/std 정규화 적용
   --runs N           : 랜덤 테스트 반복 횟수 (기본 10)
   --seed S           : 난수 시드 (기본 0)
   --half-pixel       : 역투영/보정 시 px+0.5, py+0.5 사용
   --margin-mult M    : 가장자리 margin = heatmap칸 * M (기본 2.0)
   --calib-samples K  : 보정 샘플 수 (기본 6). 0이면 보정 비활성화와 동일
   --no-calib         : 보정 비활성화
   --no-center-anchor : 보정 시 센터 앵커(중앙 포인트) 제외

사용 예:
python check.py --weights BaselinePretrained.pth.tar --frame frame.png --runs 20 \
  --imagenet-norm --letterbox --half-pixel --margin-mult 4 --calib-samples 8
"""

import os
import cv2
import argparse
import numpy as np
import torch

from model import SiameseNet, BaselineEmbeddingNet  # 프로젝트의 모델 정의 사용

# -------------------------
# 설정 상수
# -------------------------
SEARCH_SIZE = (255, 255)  # (W,H)
TEMPLATE_SIZE = (127, 127)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# -------------------------
# 유틸 함수
# -------------------------
def preprocess_rgb(img_rgb: np.ndarray, size: tuple[int, int], device: torch.device, imagenet_norm: bool) -> torch.Tensor:
    """RGB ndarray -> [1,3,H,W] float tensor, [0,1] 정규화 (+ 선택적 ImageNet 정규화)"""
    img = cv2.resize(img_rgb, size).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    if imagenet_norm:
        t = (t - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    return t


def letterbox_rgb(img_rgb: np.ndarray, target_wh=(255, 255), pad_val=(0, 0, 0)):
    """종횡비 유지 + 패딩으로 목표 크기 만들기. (out, scale, pad_x, pad_y) 반환"""
    tw, th = target_wh
    h, w = img_rgb.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh))
    pad_x = (tw - nw) // 2
    pad_y = (th - nh) // 2
    out = cv2.copyMakeBorder(resized, pad_y, th - nh - pad_y, pad_x, tw - nw - pad_x,
                             borderType=cv2.BORDER_CONSTANT, value=pad_val)
    return out, scale, pad_x, pad_y


def center_crop_template(search_rgb: np.ndarray, z_wh=(127, 127)):
    """서치 중앙에서 템플릿 크롭"""
    ZW, ZH = z_wh
    TW, TH = search_rgb.shape[1], search_rgb.shape[0]
    cx, cy = TW // 2, TH // 2
    x0, y0 = cx - ZW // 2, cy - ZH // 2
    x1, y1 = x0 + ZW, y0 + ZH
    template_rgb = search_rgb[y0:y1, x0:x1]
    return template_rgb, (x0, y0)


def random_crop_template(search_rgb: np.ndarray, z_wh=(127, 127), margin_px: int = 0):
    """가장자리 margin을 두고 랜덤 템플릿 크롭"""
    ZW, ZH = z_wh
    TW, TH = search_rgb.shape[1], search_rgb.shape[0]
    x0 = np.random.randint(margin_px, TW - ZW - margin_px + 1)
    y0 = np.random.randint(margin_px, TH - ZH - margin_px + 1)
    x1, y1 = x0 + ZW, y0 + ZH
    template_rgb = search_rgb[y0:y1, x0:x1]
    return template_rgb, (x0, y0)


def run_siamese(model: torch.nn.Module,
                template_rgb: np.ndarray,
                search_rgb: np.ndarray,
                device: torch.device,
                imagenet_norm: bool):
    """템플릿/서치 전처리 → 모델 추론 → 응답맵/피크 반환"""
    tmpl = preprocess_rgb(template_rgb, TEMPLATE_SIZE, device, imagenet_norm)
    srch = preprocess_rgb(search_rgb,   SEARCH_SIZE,  device, imagenet_norm)
    with torch.no_grad():
        resp = model(tmpl, srch)  # [1,1,h,w] 또는 [h,w]
    resp = resp.squeeze().detach().cpu().numpy()
    h, w = resp.shape
    py, px = np.unravel_index(np.argmax(resp), resp.shape)
    vmax = float(resp.max())
    return (px, py), (w, h), vmax


def map_heatmap_to_search_center(px: float, py: float, hw: tuple[int, int],
                                 half_pixel: bool = False,
                                 search_wh: tuple[int, int] = SEARCH_SIZE):
    """
    (기본) heatmap (px,py) -> 서치좌표 (sx,sy)
    센터 기준 + 칸폭. 필요 시 반픽셀 보정(px+0.5).
    """
    w, h = hw
    TW, TH = search_wh
    s_x = TW / w
    s_y = TH / h
    cx_hm = (w - 1) / 2.0
    cy_hm = (h - 1) / 2.0
    dx = (px + 0.5 if half_pixel else px) - cx_hm
    dy = (py + 0.5 if half_pixel else py) - cy_hm
    sx = dx * s_x + TW / 2.0
    sy = dy * s_y + TH / 2.0
    return sx, sy


def calibrate_linear_mapping(samples, half_pixel: bool, w_hm: int, h_hm: int,
                             use_center_anchor: bool, TW: int, TH: int):
    """
    축별 선형 보정 a,b 를 최소제곱으로 추정.
    samples: 리스트 [(px, py, exp_sx, exp_sy), ...]
    반환: (a_x, b_x, a_y, b_y), (rms_x, rms_y)
    """
    Xx, yx = [], []
    Xy, yy = [], []

    # 선택: 센터 앵커 추가 (px,py) = ((w-1)/2, (h-1)/2) -> (sx,sy)=(TW/2,TH/2)
    if use_center_anchor:
        px_c = (w_hm - 1) / 2.0 + (0.5 if half_pixel else 0.0)
        py_c = (h_hm - 1) / 2.0 + (0.5 if half_pixel else 0.0)
        sx_c = TW / 2.0
        sy_c = TH / 2.0
        Xx.append([px_c, 1.0]); yx.append(sx_c)
        Xy.append([py_c, 1.0]); yy.append(sy_c)

    for (px, py, exp_sx, exp_sy) in samples:
        px_eff = px + 0.5 if half_pixel else px
        py_eff = py + 0.5 if half_pixel else py
        Xx.append([px_eff, 1.0]); yx.append(exp_sx)
        Xy.append([py_eff, 1.0]); yy.append(exp_sy)

    Xx = np.asarray(Xx, dtype=np.float64); yx = np.asarray(yx, dtype=np.float64)
    Xy = np.asarray(Xy, dtype=np.float64); yy = np.asarray(yy, dtype=np.float64)

    # 최소제곱 해
    # 보호: 특이행렬 방지용 정규화(릿지 아주 약하게)
    reg = 1e-8
    a_b_x = np.linalg.lstsq(Xx.T @ Xx + reg*np.eye(2), Xx.T @ yx, rcond=None)[0]
    a_b_y = np.linalg.lstsq(Xy.T @ Xy + reg*np.eye(2), Xy.T @ yy, rcond=None)[0]
    a_x, b_x = float(a_b_x[0]), float(a_b_x[1])
    a_y, b_y = float(a_b_y[0]), float(a_b_y[1])

    # RMS 오차 계산
    pred_sx = Xx @ np.array([a_x, b_x])
    pred_sy = Xy @ np.array([a_y, b_y])
    rms_x = float(np.sqrt(np.mean((pred_sx - yx)**2))) if len(yx) > 0 else 0.0
    rms_y = float(np.sqrt(np.mean((pred_sy - yy)**2))) if len(yy) > 0 else 0.0

    return (a_x, b_x, a_y, b_y), (rms_x, rms_y)


def map_heatmap_to_search_linear(px: float, py: float,
                                 a_x: float, b_x: float, a_y: float, b_y: float,
                                 half_pixel: bool):
    """보정된 선형식 사용: sx = a_x * (px+δ) + b_x, sy = a_y * (py+δ) + b_y"""
    px_eff = px + 0.5 if half_pixel else px
    py_eff = py + 0.5 if half_pixel else py
    sx = a_x * px_eff + b_x
    sy = a_y * py_eff + b_y
    return sx, sy


# -------------------------
# 메인
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Siamese 가중치 경로(.pth/.tar). dict 또는 state_dict 모두 허용")
    ap.add_argument("--frame", default=None, help="테스트 프레임 경로(BGR). 없으면 데모 프레임 생성")
    ap.add_argument("--letterbox", action="store_true", help="서치 생성 시 종횡비 유지 + 패딩 사용")
    ap.add_argument("--imagenet-norm", action="store_true", help="ImageNet mean/std 정규화 적용")
    ap.add_argument("--runs", type=int, default=10, help="랜덤 테스트 반복 횟수")
    ap.add_argument("--seed", type=int, default=0, help="난수 시드")
    ap.add_argument("--half-pixel", action="store_true", help="역투영/보정 시 반픽셀 보정(px+0.5) 사용")
    ap.add_argument("--margin-mult", type=float, default=2.0, help="가장자리 margin = heatmap칸*M (기본 2.0)")
    ap.add_argument("--calib-samples", type=int, default=6, help="보정 샘플 수(K). 0이면 보정 안함")
    ap.add_argument("--no-calib", action="store_true", help="보정 강제 비활성화")
    ap.add_argument("--no-center-anchor", action="store_true", help="보정 시 센터 앵커 제외")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 모델 로드
    emb = BaselineEmbeddingNet().to(device)
    model = SiameseNet(emb).to(device)
    assert os.path.isfile(args.weights), f"가중치 파일을 찾을 수 없습니다: {args.weights}"
    ckpt = torch.load(args.weights, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 2) 프레임 준비 (RGB)
    if args.frame and os.path.isfile(args.frame):
        bgr = cv2.imread(args.frame)
        if bgr is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {args.frame}")
        frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        # 데모 프레임 생성: 단색 배경 + 사각형
        demo = np.full((480, 640, 3), 180, np.uint8)
        cv2.rectangle(demo, (270, 190), (370, 290), (200, 200, 200), -1)
        frame_rgb = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)

    # 3) 서치(255x255) 만들기
    if args.letterbox:
        search_rgb, scale, pad_x, pad_y = letterbox_rgb(frame_rgb, SEARCH_SIZE)
    else:
        search_rgb = cv2.resize(frame_rgb, SEARCH_SIZE)

    # 4) 센터 크롭 테스트 (가중치/중심 정렬 확인)
    template_rgb, (x0_c, y0_c) = center_crop_template(search_rgb, TEMPLATE_SIZE)
    (px_c, py_c), (w_hm, h_hm), vmax_c = run_siamese(model, template_rgb, search_rgb, device, args.imagenet_norm)
    exp_cx, exp_cy = (w_hm - 1) / 2.0, (h_hm - 1) / 2.0
    off_x = px_c - exp_cx
    off_y = py_c - exp_cy
    print(f"[CENTER] heatmap peak=({px_c:.1f},{py_c:.1f}) map=({w_hm},{h_hm}) "
          f"offset=({off_x:.1f},{off_y:.1f}) vmax={vmax_c:.4f}")

    # 5) margin 계산 (heatmap 칸폭 기반)
    TW, TH = SEARCH_SIZE
    cell_px = TW / w_hm
    margin_px = int(round(args.margin_mult * cell_px))

    # 6) (옵션) 축별 선형 보정(a,b) 추정
    use_calib = (not args.no_calib) and (args.calib-samples if hasattr(args, "calib-samples") else True)  # guard
    # 위 guard는 argparse 이름이 하이픈(-)이라 hasattr 대응이 어려워 대체 로직 아래에서 처리
    use_calib = (not args.no_calib) and (args.calib_samples > 0)

    a_x = b_x = a_y = b_y = None
    if use_calib:
        calib_samples = []
        # (선택) 센터 앵커 포함 여부
        use_center_anchor = (not args.no_center_anchor)

        # K회 샘플 수집 (가장자리 margin 적용)
        K = args.calib_samples
        for _ in range(K):
            tmpl_rgb, (x0, y0) = random_crop_template(search_rgb, TEMPLATE_SIZE, margin_px=margin_px)
            (px, py), (w2, h2), _ = run_siamese(model, tmpl_rgb, search_rgb, device, args.imagenet_norm)
            exp_sx = x0 + TEMPLATE_SIZE[0] // 2
            exp_sy = y0 + TEMPLATE_SIZE[1] // 2
            calib_samples.append((float(px), float(py), float(exp_sx), float(exp_sy)))

        (a_x, b_x, a_y, b_y), (rms_x, rms_y) = calibrate_linear_mapping(
            calib_samples, args.half_pixel, w_hm, h_hm, use_center_anchor, TW, TH
        )

        print(f"[CALIB] a_x={a_x:.6f}, b_x={b_x:.3f} | a_y={a_y:.6f}, b_y={b_y:.3f} "
              f"(rms_x={rms_x:.2f}px, rms_y={rms_y:.2f}px, samples={K}, center_anchor={use_center_anchor})")

    # 7) 랜덤 테스트 반복
    errs = []
    for i in range(args.runs):
        tmpl_rgb, (x0, y0) = random_crop_template(search_rgb, TEMPLATE_SIZE, margin_px=margin_px)
        (px, py), (w2, h2), vmax = run_siamese(model, tmpl_rgb, search_rgb, device, args.imagenet_norm)

        # 좌표 변환: 보정 사용 여부에 따라 분기
        if use_calib and (a_x is not None):
            sx, sy = map_heatmap_to_search_linear(px, py, a_x, b_x, a_y, b_y, args.half_pixel)
        else:
            sx, sy = map_heatmap_to_search_center(px, py, (w2, h2), half_pixel=args.half_pixel, search_wh=SEARCH_SIZE)

        # 기대 중심(서치 좌표): 템플릿 중심
        exp_sx = x0 + TEMPLATE_SIZE[0] // 2
        exp_sy = y0 + TEMPLATE_SIZE[1] // 2

        err = float(np.hypot(sx - exp_sx, sy - exp_sy))
        errs.append(err)
        print(f"[RANDOM#{i+1}] peak_s=({int(round(sx))},{int(round(sy))}) exp=({exp_sx},{exp_sy}) "
              f"err={err:.2f}px vmax={vmax:.4f}")

    if len(errs) > 0:
        print(f"[STATS] runs={len(errs)}  mean_err={np.mean(errs):.2f}px  std={np.std(errs):.2f}px  "
              f"(cell≈{cell_px:.2f}px, margin≈{margin_px}px)  half_pixel={args.half_pixel}  "
              f"calib={'on' if use_calib else 'off'}")

    # 참고: letterbox 사용 시 원본 좌표로 되돌리려면 (필요할 때)
    # orig_x = int((sx - pad_x) / scale)
    # orig_y = int((sy - pad_y) / scale)

if __name__ == "__main__":
    main()
