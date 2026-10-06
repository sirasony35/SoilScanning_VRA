# -*- coding: utf-8 -*-
"""
변량시비 이앙기(ADCU) 처방맵 배포 CSV 생성기.

규격: 변량시비이앙기/변량시비이앙기_처방맵배포파일_데이터정의서_260917_v1.xlsx (2026-09-17 v1)
  · 형식  : CSV, UTF-8(BOM), CRLF
  · 구성  : ① '#' 메타 헤더 블록 → ② 빈 줄 → ③ 컬럼 헤더 → ④ 그리드 레코드 N행
  · 좌표계: WGS84 십진수
  · 꼭짓점: v1~v7 (최대 7개)

토양 포인트 → 시비 공식 → 처방량 산출까지는 main_grid_vra.py 로직을 그대로 따르고,
마지막 출력만 ADCU 규격 CSV로 바꾼 것이다.
"""
import os
import re
import math
import glob
import zipfile
import datetime
import binascii

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import affinity

import main_grid_vra as M
from fertilizer_calculator import FertilizerCalculator

# ======================================================
# 0. 설정
# ======================================================
DATA_FOLDER = "real_data"
RESULT_ROOT = "result_adcu"
FIELD_NAME = "SM-1-1"              # 대상 필지 (real_data 안의 core name)
GRID_SIZES = [5, 10, 20]           # 생성할 격자 해상도 (m)

MAX_VERTEX = 7                     # 규격 상한 (v1~v7)
COORD_DECIMALS = 7                 # 좌표 소수 자릿수 (7 ≈ 1.1cm)

# --- 시비 조건 (main_grid_vra.py와 동일) ---
CROP_TYPE = 'soybean'
TARGET_YIELD = 480
BASAL_RATIO = 100
SOIL_TEXTURE = '사질'
MIN_N_REQUIREMENT = 2.0
FERTILIZER_N_CONTENT = 0.08
NUM_CLASSES = 5                    # 처방 등급 수 (예시 CSV도 5단계)

FIELD_TARGET_KG = {"SM-1-1": 1500, "SM-1-2": 1500, "SM-1-3": 1500,
                   "SM-2-2": 1360, "SM-2-3": 1340}
FIELD_FUNGICIDE_KG = {"SM-1-1": 120, "SM-1-2": 120, "SM-1-3": 120,
                      "SM-2-2": 120, "SM-2-3": 120}

# --- 메타 헤더 고정값 (※ 실배포 시 서버가 채우는 값. 여기서는 형식 확인용 임시값) ---
META_FIXED = {
    'vehicle_id':     'ADCU-2000731',
    'account_id':     'ACC-DAEDONG-0001',
    'fert_id':        'PN-MP300',
    'fert_name':      '풍농 명품300 (측조용)',
    'fert_npk':       '30-7-9',
    'fert_density':   '0.90',
    'bag_kg':         20,
    'crop':           '논콩',          # real_data 기준 (이앙기 실배포는 '벼')
    'boundary_id':    'soilscan-260528',
    'step_table_ver': 'HSV-실측-260829',
}

COLUMNS = (['grid_id', 'center_lat', 'center_lon', 'rate_kg_ha',
            'area_m2', 'clipped', 'vertex_count']
           + [f'v{i}_{c}' for i in range(1, MAX_VERTEX + 1) for c in ('lat', 'lon')])


# ======================================================
# 1. 꼭짓점 상한 처리
# ======================================================
def cap_vertices(poly, max_v=MAX_VERTEX):
    """
    꼭짓점이 max_v를 넘는 폴리곤을 단순화해 상한 이하로 줄인다.
    경계 클리핑 셀은 경계선을 그대로 따라가 꼭짓점이 수십 개까지 늘어나므로 필수.
    반환: (폴리곤, 단순화 적용 여부)
    """
    n = len(poly.exterior.coords) - 1
    if n <= max_v:
        return poly, False

    tol = 0.05
    for _ in range(60):
        s = poly.simplify(tol, preserve_topology=True)
        if (not s.is_empty) and s.geom_type == 'Polygon':
            if len(s.exterior.coords) - 1 <= max_v:
                return s, True
        tol *= 1.3
    # 최후 수단: 최소회전사각형(4각)
    return poly.minimum_rotated_rectangle, True


# ======================================================
# 2. 처방량 산출 (main_grid_vra.py 로직 준용)
# ======================================================
def build_prescription(soil_path, boundary_path, base_name, grid_size):
    """단일 필지·단일 해상도 처방 격자 생성. 반환: (격자 gdf, 회전각(도), 경계 geom)"""
    # --- 입력 로드 ---
    soil_in = f"zip://{soil_path}".replace("\\", "/")
    bound_in = f"zip://{boundary_path}".replace("\\", "/")

    points_gdf = None
    for enc in ['euc-kr', 'cp949', 'latin1', 'utf-8']:
        try:
            points_gdf = gpd.read_file(soil_in, encoding=enc)
            break
        except Exception:
            continue
    if points_gdf is None:
        points_gdf = gpd.read_file(soil_in)
    boundary_gdf = gpd.read_file(bound_in)

    points_meter = M.fix_coordinate_system(points_gdf, "토양점")
    boundary_meter = M.fix_coordinate_system(boundary_gdf, "경계")

    # --- 컬럼 정제 ---
    points_meter = points_meter.rename(
        columns={c: re.sub(r'[^A-Z0-9]', '', c.upper()) for c in points_meter.columns})
    points_meter = points_meter.loc[:, ~points_meter.columns.duplicated()]
    if 'GEOMETRY' in points_meter.columns:
        points_meter = points_meter.rename(columns={'GEOMETRY': 'geometry'})
    points_meter = points_meter.set_geometry('geometry')

    exclude = ['ID', 'LATITUDE', 'LONGITUDE', 'COUNTRATE', 'geometry']
    for col in points_meter.columns:
        if col not in exclude:
            points_meter[col] = pd.to_numeric(points_meter[col], errors='coerce')
            if points_meter[col].isnull().sum() > 0:
                points_meter[col] = pd.to_numeric(
                    points_meter[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0],
                    errors='coerce')
            points_meter[col] = points_meter[col].fillna(0)

    numeric_cols = points_meter.select_dtypes(include=[np.number]).columns.tolist()
    analysis_cols = [c for c in numeric_cols if c not in exclude]
    for ess in ['OM', 'SI', 'PH', 'P', 'K', 'MG']:
        if ess in points_meter.columns and ess not in analysis_cols:
            analysis_cols.append(ess)

    # --- 회전 정렬 격자 ---
    boundary_geom = boundary_meter.union_all()
    rotation_angle = M.get_main_angle(boundary_geom)
    centroid = boundary_geom.centroid
    rotated = affinity.rotate(boundary_geom, -rotation_angle, origin=centroid)
    xmin, ymin, xmax, ymax = rotated.bounds

    polys = [Polygon([(x, y), (x + grid_size, y),
                      (x + grid_size, y + grid_size), (x, y + grid_size)])
             for x in np.arange(xmin, xmax, grid_size)
             for y in np.arange(ymin, ymax, grid_size)]
    temp_grid = gpd.GeoDataFrame({'geometry': polys}, crs="EPSG:5179")
    temp_grid['geometry'] = temp_grid.geometry.apply(
        lambda g: affinity.rotate(g, rotation_angle, origin=centroid))
    grid = gpd.clip(temp_grid, boundary_meter).reset_index(drop=True)
    grid = grid[~grid.geometry.is_empty & grid.geometry.notna()].copy()
    grid = grid[grid.geometry.geom_type == 'Polygon'].copy()
    grid = grid.reset_index(drop=True)
    grid['grid_id'] = grid.index + 1

    # --- 셀별 토양값 + 결측 IDW 보간 ---
    joined = gpd.sjoin(grid, points_meter, how="inner", predicate="intersects")
    stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()
    grid = grid.merge(stats, on='grid_id', how='left')
    if grid.crs is None:
        grid.set_crs("EPSG:5179", inplace=True)

    critical = [c for c in (['OM', 'SI'] if CROP_TYPE == 'rice' else ['OM'])
                if c in grid.columns]
    if critical:
        grid = M.fill_nan_by_idw(grid, critical, k=5, power=2)
    grid[analysis_cols] = grid[analysis_cols].fillna(0)

    # --- 시비량 계산 ---
    grid = FertilizerCalculator(
        grid, crop_type=CROP_TYPE, target_yield=TARGET_YIELD,
        basal_ratio=BASAL_RATIO, fertilizer_n_content=FERTILIZER_N_CONTENT,
        soil_texture=SOIL_TEXTURE, min_n_limit=MIN_N_REQUIREMENT).execute()

    # --- 목표량 + 살균제 스케일링 ---
    total_sys = grid['F_Total'].sum()
    base_kg = FIELD_TARGET_KG.get(base_name, 0) or total_sys
    total_mixed = base_kg + FIELD_FUNGICIDE_KG.get(base_name, 0)
    mix_ratio = (total_mixed / total_sys) if total_sys > 0 else 1.0
    dose = grid['F_Need_10a'] * 10 * mix_ratio       # kg/ha

    # --- 5등급 분류 (예시 CSV도 5단계) ---
    if dose.nunique() > NUM_CLASSES:
        _, bins = pd.cut(dose, bins=NUM_CLASSES, retbins=True, duplicates='drop')
        labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        dose = pd.cut(dose, bins=bins, labels=labels,
                      include_lowest=True).astype(float)
    grid['rate_kg_ha'] = dose.round().astype(int)     # 규격: INT

    azimuth = (90.0 - rotation_angle) % 180.0          # 정북 기준 방위각
    return grid, azimuth, boundary_geom, total_mixed


# ======================================================
# 3. ADCU 규격 CSV 출력
# ======================================================
def export_adcu_csv(grid, azimuth, base_name, grid_size, out_path):
    """규격서 v1 형식으로 CSV 작성. 반환: 통계 dict"""
    g5179 = grid.to_crs(5179)
    g4326 = grid.to_crs(4326)

    records, n_simplified, area_before, area_after = [], 0, 0.0, 0.0
    vtx_hist = {}

    for idx in range(len(grid)):
        poly_m = g5179.geometry.iloc[idx]
        area_before += poly_m.area

        poly_m, simplified = cap_vertices(poly_m)
        if simplified:
            n_simplified += 1
        area_m2 = poly_m.area
        area_after += area_m2

        # 단순화 결과를 WGS84로 (단순화 안 됐으면 원본 그대로)
        poly_w = (gpd.GeoSeries([poly_m], crs=5179).to_crs(4326).iloc[0]
                  if simplified else g4326.geometry.iloc[idx])

        coords = list(poly_w.exterior.coords)[:-1]        # 닫는 중복점 제거
        vcount = len(coords)
        vtx_hist[vcount] = vtx_hist.get(vcount, 0) + 1

        cen = poly_w.centroid
        clipped = 1 if area_m2 < grid_size ** 2 * 0.999 else 0

        row = {
            'grid_id': int(grid['grid_id'].iloc[idx]),
            'center_lat': round(cen.y, COORD_DECIMALS),
            'center_lon': round(cen.x, COORD_DECIMALS),
            'rate_kg_ha': int(grid['rate_kg_ha'].iloc[idx]),
            'area_m2': round(area_m2, 1),
            'clipped': clipped,
            'vertex_count': vcount,
        }
        for i in range(MAX_VERTEX):
            if i < vcount:
                lon, lat = coords[i]
                row[f'v{i+1}_lat'] = round(lat, COORD_DECIMALS)
                row[f'v{i+1}_lon'] = round(lon, COORD_DECIMALS)
            else:
                row[f'v{i+1}_lat'] = ''
                row[f'v{i+1}_lon'] = ''
        records.append(row)

    df = pd.DataFrame(records, columns=COLUMNS)

    # --- 총량 · 대체값 (최종 폴리곤 기준으로 내부 정합) ---
    total_kg = float((df['rate_kg_ha'] * df['area_m2']).sum() / 10000.0)
    total_ha = float(df['area_m2'].sum() / 10000.0)
    avg_rate = int(round(total_kg / total_ha)) if total_ha > 0 else 0
    bags = int(math.ceil(total_kg / META_FIXED['bag_kg']))

    # --- 본문(컬럼헤더 + 레코드) 먼저 만들고 CRC32 계산 ---
    body_lines = [','.join(COLUMNS)]
    for r in df.itertuples(index=False):
        body_lines.append(','.join('' if v == '' else str(v) for v in r))
    body = '\r\n'.join(body_lines) + '\r\n'
    crc = binascii.crc32(body.encode('utf-8')) & 0xFFFFFFFF

    today = datetime.datetime.now().strftime('%Y%m%d')
    meta = [
        ('map_id', f'RX-{base_name}-{today}-01', '(처방맵 ID = 배포 식별자)'),
        ('work_id', '= map_id + sessionSeq(+uploadSeq)', '(작업 식별: 세션순번은 ADCU 채번)'),
        ('field_id', base_name, '필지명'),
        ('vehicle_id', META_FIXED['vehicle_id'], '대상 차량(ADCU) ID'),
        ('account_id', META_FIXED['account_id'], '계정 ID'),
        ('fert_id', META_FIXED['fert_id'], '비료 ID'),
        ('fert_name', META_FIXED['fert_name'], ''),
        ('fert_npk', META_FIXED['fert_npk'], '비료 비율 N-P-K(%)'),
        ('fert_density', META_FIXED['fert_density'], '비중'),
        ('bag_kg', META_FIXED['bag_kg'], '포장 단위(kg)'),
        ('crop', META_FIXED['crop'], ''),
        ('grid_size_m', grid_size, ''),
        ('grid_azimuth_deg', round(azimuth, 1), '격자 회전각(정북 아님)'),
        ('boundary_id', META_FIXED['boundary_id'], '처방 경계'),
        ('grid_count_total', len(df), '필지 전체 셀 수'),
        ('total_kg', round(total_kg, 1), '총 살포량(목표)'),
        ('avg_rate_kg_ha', avg_rate, '평균 살포량 = 대체(fallback) 살포용'),
        ('bags', bags, '총 포대수'),
        ('step_table_ver', META_FIXED['step_table_ver'], '단수↔kg 조견표 버전(실적 해석용)'),
        ('crc32', f'0x{crc:08X}', '파일 무결성 검사 기준값(본문 CRC32)'),
        ('coord_system', 'WGS84', ''),
    ]
    head_lines = ['# 변량시비 처방맵 배포 데이터 (서버 → ADCU)']
    for k, v, note in meta:
        head_lines.append(f'# {k},{v},  {note}'.rstrip() if note else f'# {k},{v}')

    text = '\r\n'.join(head_lines) + '\r\n\r\n' + body

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(b'\xef\xbb\xbf')                 # UTF-8 BOM
        f.write(text.encode('utf-8'))

    return {
        'cells': len(df), 'total_kg': total_kg, 'total_ha': total_ha,
        'avg_rate': avg_rate, 'bags': bags, 'crc': f'0x{crc:08X}',
        'bytes': os.path.getsize(out_path),
        'n_simplified': n_simplified,
        'area_before': area_before, 'area_after': area_after,
        'vtx_hist': vtx_hist, 'clipped': int(df['clipped'].sum()),
        'rates': sorted(df['rate_kg_ha'].unique().tolist()),
    }


# ======================================================
# 4. 메인
# ======================================================
def main():
    soil = os.path.join(DATA_FOLDER, f"{FIELD_NAME}_Shapefile.zip")
    bound = os.path.join(DATA_FOLDER, f"{FIELD_NAME}_Boundary.zip")
    for p in (soil, bound):
        if not os.path.exists(p):
            print(f"[오류] 파일 없음: {p}")
            return

    print("=" * 72)
    print(f" 변량시비 이앙기(ADCU) 처방맵 CSV 생성 — {FIELD_NAME}")
    print("=" * 72)

    summary = []
    for gs in GRID_SIZES:
        print(f"\n▶ {gs}m 격자")
        grid, azimuth, boundary_geom, target_kg = build_prescription(
            soil, bound, FIELD_NAME, gs)

        out_dir = os.path.join(RESULT_ROOT, FIELD_NAME)
        out_path = os.path.join(out_dir, f"{FIELD_NAME}_{gs}m_ADCU.csv")
        st = export_adcu_csv(grid, azimuth, FIELD_NAME, gs, out_path)

        area_err = (st['area_after'] - st['area_before']) / st['area_before'] * 100
        print(f"  - 셀 수        : {st['cells']:,}개 "
              f"(경계 클리핑 {st['clipped']}개, {st['clipped']/st['cells']*100:.1f}%)")
        print(f"  - 처방 단계    : {st['rates']} kg/ha")
        print(f"  - 총 살포량    : {st['total_kg']:,.1f} kg "
              f"(목표 {target_kg:,.0f} kg) / 평균 {st['avg_rate']} kg/ha / {st['bags']}포")
        print(f"  - 꼭짓점 단순화: {st['n_simplified']}개 셀 "
              f"({st['n_simplified']/st['cells']*100:.1f}%), 면적 변화 {area_err:+.2f}%")
        print(f"  - 꼭짓점 분포  : " +
              ", ".join(f"{k}개:{v}셀" for k, v in sorted(st['vtx_hist'].items())))
        print(f"  - 파일 크기    : {st['bytes']:,} B ({st['bytes']/1024:.1f} KB), "
              f"셀당 {st['bytes']/st['cells']:.0f} B")
        print(f"  - CRC32        : {st['crc']}")
        print(f"  → {out_path}")
        summary.append((gs, st))

    print("\n" + "=" * 72)
    print(" 요약 — 시비기 업체 전달용")
    print("=" * 72)
    print(f"{'격자':>6} | {'셀 수':>8} | {'파일 크기':>12} | {'셀당':>7} | {'1ha 환산':>10}")
    print("-" * 72)
    for gs, st in summary:
        per_ha = st['bytes'] / st['total_ha']
        print(f"{gs:>4}m | {st['cells']:>8,} | {st['bytes']:>9,} B | "
              f"{st['bytes']/st['cells']:>5.0f} B | {per_ha/1024:>7.1f} KB/ha")
    print(f"\n[완료] {os.path.abspath(RESULT_ROOT)}")


if __name__ == "__main__":
    main()
