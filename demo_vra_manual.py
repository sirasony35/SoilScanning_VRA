# -*- coding: utf-8 -*-
"""
시연용 변량시비(VRA) 처방맵 생성기 — 토양 분석 데이터 없이 '임의의 비료량'으로 작성.

기존 main_grid_vra.py 는 토양 포인트(OM 등) → 시비 공식 → 처방량 순서지만,
이 스크립트는 **필지 경계만** 입력받아
  1) 살포폭(=격자) 기준으로 필지를 분할하고
  2) 정해진 등급 비율(예: 1:2:3)로 존을 배치한 뒤
  3) 필지 총 살포량(예: 10포대 = 200kg)에 정확히 맞게 처방량을 역산
하여 ISOXML / SHP / CSV / DJI TIF 를 출력한다.

출력 함수(export_isoxml, export_csv, export_dji_tif)는 main_grid_vra.py 것을 그대로 재사용하므로
FMS(대동 FieldFusion)에 올라가는 포맷·단위 규약(kg/ha ×100 → mg/m²)은 기존과 100% 동일하다.
"""

import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity

import main_grid_vra as M

# ======================================================
# 0. 환경 설정
# ======================================================
BOUNDARY_PATH = r"test_data/간척지_비료시연.zip"   # 경계 폴리곤 (zip/shp)
RESULT_ROOT = "result_demo"
BASE_NAME = "간척지_비료시연"
TASK_NAME_ASCII = "GANCHEOKDEMO"      # ISOXML TSK 이름(영문/숫자만 유효)

WORK_WIDTH = 32.0                      # 살포폭 (m) — 패스 폭이자 격자 한 변
TOTAL_FERT_KG = 200.0                  # 필지 총 살포량 (10포대 × 20kg)

# 등급 상대 비율. 길이 = 등급 수. 1:2:3 이면 최고등급이 최저등급의 3배.
ZONE_WEIGHTS = [1.0, 2.0, 3.0]

# 존 배치 방식
#   'shift'  : 패스마다 등급을 한 칸씩 밀어 배치 → 주행 중 등급이 계속 바뀜 (시연 추천)
#   'stripe' : 패스 전체를 한 등급으로 → 지면에 3개 띠로 남음
#   'checker': 인접 칸이 항상 다른 등급
ZONE_PATTERN = 'shift'

# 자투리 셀 병합 기준: (셀 면적 / 온전한 셀 면적)이 이 값 미만이면 이웃 패스에 흡수
SLIVER_RATIO = 0.30

EXPORT_ISOXML = True
EXPORT_SHP = True
EXPORT_CSV = True
EXPORT_DJI = True


# ======================================================
# 1. 격자 생성 (주행방향 = 필지 장축)
# ======================================================
def build_pass_grid(boundary_gdf, work_width, sliver_ratio):
    """
    필지 장축을 주행방향으로 잡고 격자를 생성한다.
      - 회전 후 x축 = 주행방향(장축), y축 = 살포폭 방향(패스)
      - 살포폭 방향의 자투리 패스(폭이 work_width의 sliver_ratio 미만)는
        인접 패스에 흡수시켜 실제로 주행 불가능한 띠가 생기지 않게 한다.
    반환: (격자 GeoDataFrame[seg, pass_no, geometry], boundary_geom, 회전각, 치수 정보)
    """
    boundary_geom = boundary_gdf.union_all()
    angle = M.get_main_angle(boundary_geom)
    origin = boundary_geom.centroid

    rotated = affinity.rotate(boundary_geom, -angle, origin=origin)
    xmin, ymin, xmax, ymax = rotated.bounds
    length_drive = xmax - xmin      # 주행방향 길이
    length_pass = ymax - ymin       # 살포폭 방향 길이

    # --- 살포폭 방향(패스) 경계: 자투리는 마지막 패스에 흡수 ---
    n_pass = max(1, int(round(length_pass / work_width)))
    y_edges = [ymin + i * work_width for i in range(n_pass)]
    if (length_pass - n_pass * work_width) / work_width >= sliver_ratio:
        y_edges.append(ymin + n_pass * work_width)   # 자투리가 충분히 크면 독립 패스로
    y_edges.append(ymax)

    # --- 주행방향 구간 경계: 자투리는 그대로 짧은 구간으로 둠(주행에는 지장 없음) ---
    n_seg_full = int(length_drive // work_width)
    x_edges = [xmin + i * work_width for i in range(n_seg_full + 1)]
    if x_edges[-1] < xmax - 1e-6:
        if (xmax - x_edges[-1]) / work_width < sliver_ratio:
            x_edges[-1] = xmax          # 너무 짧으면 직전 구간에 흡수
        else:
            x_edges.append(xmax)

    records = []
    for si in range(len(x_edges) - 1):
        for pi in range(len(y_edges) - 1):
            x0, x1 = x_edges[si], x_edges[si + 1]
            y0, y1 = y_edges[pi], y_edges[pi + 1]
            poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            records.append({'seg': si, 'pass_no': pi, 'geometry': poly})

    grid = gpd.GeoDataFrame(records, crs=boundary_gdf.crs)
    grid['geometry'] = grid.geometry.apply(lambda g: affinity.rotate(g, angle, origin=origin))
    grid = gpd.clip(grid, boundary_gdf).reset_index(drop=True)
    grid = grid[~grid.geometry.is_empty & grid.geometry.notna()].copy()
    grid = grid[grid.geometry.area > 1.0].copy()          # 클립 부스러기 제거
    grid = grid.sort_values(['pass_no', 'seg']).reset_index(drop=True)

    dims = {
        'angle': angle,
        'length_drive': length_drive,
        'length_pass': length_pass,
        'n_pass': len(y_edges) - 1,
        'n_seg': len(x_edges) - 1,
        'pass_widths': [round(y_edges[i + 1] - y_edges[i], 1) for i in range(len(y_edges) - 1)],
        'seg_lengths': [round(x_edges[i + 1] - x_edges[i], 1) for i in range(len(x_edges) - 1)],
    }
    return grid, boundary_geom, dims


# ======================================================
# 2. 존(등급) 배치
# ======================================================
def assign_zones(grid, n_zones, pattern):
    """등급 인덱스(0=최저 … n-1=최고)를 각 셀에 배치."""
    if pattern == 'stripe':
        idx = grid['pass_no'] % n_zones
    elif pattern == 'checker':
        idx = (grid['seg'] + grid['pass_no'] * 2) % n_zones
    else:  # 'shift' — 패스마다 한 칸씩 밀어 주행 중 등급이 계속 바뀌게
        idx = (grid['seg'] + grid['pass_no']) % n_zones
    grid['ZONE_IDX'] = idx.astype(int)
    return grid


# ======================================================
# 3. 총량 고정 역산
# ======================================================
def solve_rates(grid, weights, total_kg):
    """
    Σ(처방량[kg/ha] × 면적[ha]) = total_kg 을 만족하면서
    등급 간 비율이 정확히 weights 가 되는 처방량을 구한다.
        rate_z = k × weight_z,  k = total_kg / Σ(weight_z × area_ha_z)
    """
    grid['Area_sqm'] = grid.geometry.area
    area_ha = grid['Area_sqm'] / 10000.0

    denom = float((area_ha * grid['ZONE_IDX'].map(lambda i: weights[i])).sum())
    if denom <= 0:
        raise ValueError("면적이 0입니다. 경계 데이터를 확인하세요.")

    k = total_kg / denom
    rates = [round(k * w, 2) for w in weights]

    grid['DOSE'] = grid['ZONE_IDX'].map(lambda i: rates[i])
    grid['F_Total'] = (grid['DOSE'] * area_ha).round(3)
    grid['PRODUCT'] = grid['ZONE_IDX'] + 1
    grid['ZONE'] = np.arange(1, len(grid) + 1)
    grid['DOSE_UNIT'] = 'kg/ha'
    return grid, rates


# ======================================================
# 4. 출력
# ======================================================
def export_all(grid, boundary_gdf, boundary_geom, out_dir, file_prefix):
    os.makedirs(out_dir, exist_ok=True)

    boundary_gdf.to_crs(epsg=4326).to_file(
        os.path.join(out_dir, f"{BASE_NAME}_Boundary.shp"), encoding='utf-8')

    if EXPORT_SHP:
        shp_cols = ['DOSE', 'ZONE', 'DOSE_UNIT', 'PRODUCT', 'geometry']
        output_shp = os.path.join(out_dir, f"{file_prefix}_Result.shp")
        grid[shp_cols].to_crs(epsg=4326).to_file(output_shp, encoding='utf-8')
        print(f"    - SHP 저장 완료: {os.path.basename(output_shp)}")

        zip_path = os.path.join(out_dir, f"{file_prefix}_Result_SHP.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                f = output_shp.replace('.shp', ext)
                if os.path.exists(f):
                    zf.write(f, os.path.basename(f))
        print(f"    - SHP ZIP 생성 완료: {os.path.basename(zip_path)}")

    if EXPORT_CSV:
        M.export_csv(grid.copy(), out_dir, file_prefix)

    if EXPORT_DJI:
        M.export_dji_tif(grid.copy(), out_dir, file_prefix, rate_col='DOSE')

    if EXPORT_ISOXML:
        zip_out = M.export_isoxml(grid.copy(), boundary_geom, out_dir,
                                  task_name=TASK_NAME_ASCII, rate_col='DOSE')
        if zip_out:
            print(f"    - ISOXML GRD ZIP 생성: {os.path.basename(zip_out)}")


# ======================================================
# 5. 메인
# ======================================================
def main():
    print("=" * 60)
    print(" 시연용 변량시비 처방맵 생성 (토양분석 없이 임의 배분)")
    print("=" * 60)

    src = BOUNDARY_PATH
    if src.lower().endswith('.zip'):
        src = f"zip://{BOUNDARY_PATH}"
    boundary = gpd.read_file(src)
    if boundary.crs is None:
        boundary.set_crs(epsg=4326, inplace=True)
    boundary = boundary.to_crs(epsg=5179)

    field_area = boundary.union_all().area
    print(f"\n[필지] {BASE_NAME}")
    print(f"  - 면적       : {field_area:,.1f} m²  ({field_area / 10000:.4f} ha)")

    grid, boundary_geom, dims = build_pass_grid(boundary, WORK_WIDTH, SLIVER_RATIO)
    print(f"  - 주행방향각 : {dims['angle']:.1f}°  (장축 {dims['length_drive']:.1f}m, "
          f"살포폭방향 {dims['length_pass']:.1f}m)")
    print(f"  - 패스 구성  : {dims['n_pass']}개 패스, 폭 {dims['pass_widths']} m")
    print(f"  - 주행 구간  : {dims['n_seg']}개 구간, 길이 {dims['seg_lengths']} m")
    print(f"  - 처방 셀 수 : {len(grid)}칸")

    n_zones = len(ZONE_WEIGHTS)
    grid = assign_zones(grid, n_zones, ZONE_PATTERN)
    grid, rates = solve_rates(grid, ZONE_WEIGHTS, TOTAL_FERT_KG)

    print(f"\n[등급] 비율 {':'.join(str(int(w)) if float(w).is_integer() else str(w) for w in ZONE_WEIGHTS)}"
          f" / 배치 '{ZONE_PATTERN}'")
    summary = (grid.groupby('ZONE_IDX')
               .agg(셀수=('ZONE', 'count'),
                    면적_m2=('Area_sqm', 'sum'),
                    처방량_kgha=('DOSE', 'first'),
                    투입량_kg=('F_Total', 'sum'))
               .reset_index())
    summary['ZONE_IDX'] = summary['ZONE_IDX'].map(lambda i: f"{i + 1}등급")
    for _, r in summary.iterrows():
        print(f"  - {r['ZONE_IDX']}: {r['처방량_kgha']:7.2f} kg/ha × {r['면적_m2']:7.1f} m² "
              f"({int(r['셀수'])}칸) = {r['투입량_kg']:6.2f} kg")

    total = grid['F_Total'].sum()
    print(f"  ------------------------------------------------")
    print(f"  총 투입량   : {total:,.2f} kg  (목표 {TOTAL_FERT_KG:,.1f} kg, "
          f"{total / 20:.2f}포대)")
    print(f"  필지 평균   : {total / (field_area / 10000):,.1f} kg/ha")

    print(f"\n[배치도] 행=패스(주행 라인), 열=주행 구간")
    pivot = grid.pivot_table(index='pass_no', columns='seg', values='PRODUCT', aggfunc='first')
    for pi in pivot.index:
        cells = " ".join(
            f"{int(pivot.loc[pi, s])}등급" if pd.notna(pivot.loc[pi, s]) else "  -  "
            for s in pivot.columns)
        print(f"  패스{pi + 1} │ {cells}")

    file_prefix = f"{BASE_NAME}_{int(WORK_WIDTH)}mx{int(WORK_WIDTH)}m"
    out_dir = os.path.join(RESULT_ROOT, BASE_NAME, file_prefix)
    print(f"\n[출력] {out_dir}")
    export_all(grid, boundary, boundary_geom, out_dir, file_prefix)

    print(f"\n[완료] {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
