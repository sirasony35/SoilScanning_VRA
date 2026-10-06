# -*- coding: utf-8 -*-
"""
호산비전 적합 비료별 벼 변량시비 처방맵 + 총량 비교 (SM-1-1).

- 벼(rice) 기준, 유기물(OM)·유효규산(SI) RDA 시비공식 사용.
  · SM-1-1 토양데이터에 SI 컬럼이 없어 SI는 기본값(150)으로 전 셀 동일 → 변량은 OM으로만 발생.
- 순수 질소(N) 요구량은 비료와 무관하게 동일. 비료별 N 함량(%)으로 나눠 '실제 살포 비료량(kg/ha)'을 산출.
  → 같은 논에 같은 N을 주더라도 저함량 비료일수록 물리적 살포량·총량이 커진다 (PM 하이코트 23% 질문의 핵심).
- 비료마다 처방맵 PNG(5등급) + 총 필요량/포대수/기계범위 판정을 이미지에 표기.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import affinity

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

import main_grid_vra as M
from fertilizer_calculator import FertilizerCalculator

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===== 설정 =====
DATA_FOLDER = "real_data"
FIELD = "SM-1-1"
GRID_SIZE = 10                 # m
RESULT_DIR = os.path.join("result_fertilizer", FIELD)
MAP_DIR = os.path.join(RESULT_DIR, "maps")

CROP = 'rice'
TARGET_YIELD = 480             # 벼 목표수량 kg/10a. 공식: ≥480 tier (N=9.14-0.109*OM+0.020*SI, 상한13)
BASAL_RATIO = 100              # 올코팅/측조 완효성은 1회 전량 시비
MIN_N = 2.0
NUM_CLASSES = 5
BAG_KG = 20

# 기계 시비량 범위 (SRS): 12~90 kg/10a = 120~900 kg/ha
MACHINE_MIN_KGHA = 120
MACHINE_MAX_KGHA = 900

# 호산 적합 비료 (측조적합성 ◎ 올코팅 10종 + ○ N코팅 15종)
FERTILIZERS = [
    # id, 제조사, 제품명, N, P, K, 그룹, 적용차수
    ("F01", "팜한농",   "광분해 한번에측조",     32, 7, 7,  "올코팅", 1),
    ("F02", "팜한농",   "광분해 한번에측조스피드", 32, 7, 7,  "올코팅", 1),
    ("F03", "누보",     "올코팅31",             31, 6, 8,  "올코팅", 1),
    ("F04", "누보",     "하이코트",             23, 5, 12, "올코팅", 1),   # ★ PM 질문 대상
    ("F05", "누보",     "올코팅 터보",           31, 6, 8,  "올코팅", 1),
    ("F06", "조비",     "단번에올코팅",          30, 8, 7,  "올코팅", 1),
    ("F07", "풍농",     "올코팅한포로",          32, 6, 8,  "올코팅", 1),
    ("F08", "남해화학", "오래가올원",            30, 8, 8,  "올코팅", 1),
    ("F09", "KG",       "미생물올코팅",          30, 5, 7,  "올코팅", 1),
    ("F10", "한국협화", "땅심올코팅",            30, 6, 9,  "올코팅", 1),
    ("F11", "팜한농",   "롱스타K스피드",         19, 10, 10, "N코팅", 2),
    ("F12", "팜한농",   "롱스타K플러스",         22, 7, 10, "N코팅", 2),
    ("F13", "팜한농",   "롱스타K",              19, 10, 10, "N코팅", 2),
    ("F14", "조비",     "단한번24",             24, 7, 9,  "N코팅", 2),
    ("F15", "조비",     "단한번",               18, 7, 9,  "N코팅", 2),
    ("F16", "풍농",     "명품300",              30, 7, 9,  "N코팅", 2),
    ("F17", "풍농",     "측조골드24",           24, 6, 8,  "N코팅", 2),
    ("F18", "풍농",     "측조로870",            18, 7, 10, "N코팅", 2),
    ("F19", "남해",     "골드측조",             28, 8, 9,  "N코팅", 2),
    ("F20", "남해",     "신세대22",             22, 7, 7,  "N코팅", 2),
    ("F21", "KG",       "측조로한번만",          22, 7, 9,  "N코팅", 2),
    ("F22", "KG",       "하나로",               18, 7, 8,  "N코팅", 2),
    ("F23", "한국협화", "땅심측조짱",            28, 6, 7,  "N코팅", 2),
    ("F24", "한국협화", "측조30",               30, 6, 7,  "N코팅", 2),
    ("F25", "한국협화", "지속N30",              30, 6, 7,  "N코팅", 2),
]

CLASS_COLORS = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']  # 저→고


# ===== 1. 격자 + 토양 + 순수 N =====
def build_base_grid():
    import re
    soil = f"zip://{DATA_FOLDER}/{FIELD}_Shapefile.zip"
    bnd = f"zip://{DATA_FOLDER}/{FIELD}_Boundary.zip"

    pts = None
    for enc in ['euc-kr', 'cp949', 'latin1', 'utf-8']:
        try:
            pts = gpd.read_file(soil, encoding=enc); break
        except Exception:
            continue
    if pts is None:
        pts = gpd.read_file(soil)
    bgdf = gpd.read_file(bnd)

    pts = M.fix_coordinate_system(pts, "토양점")
    bmeter = M.fix_coordinate_system(bgdf, "경계")
    pts = pts.rename(columns={c: re.sub(r'[^A-Z0-9]', '', c.upper()) for c in pts.columns})
    pts = pts.loc[:, ~pts.columns.duplicated()]
    if 'GEOMETRY' in pts.columns:
        pts = pts.rename(columns={'GEOMETRY': 'geometry'})
    pts = pts.set_geometry('geometry')
    if 'OM' in pts.columns:
        pts['OM'] = pd.to_numeric(pts['OM'], errors='coerce').fillna(0)

    boundary_geom = bmeter.union_all()
    ang = M.get_main_angle(boundary_geom)
    cen = boundary_geom.centroid
    rot = affinity.rotate(boundary_geom, -ang, origin=cen)
    xmin, ymin, xmax, ymax = rot.bounds
    polys = [Polygon([(x, y), (x + GRID_SIZE, y), (x + GRID_SIZE, y + GRID_SIZE), (x, y + GRID_SIZE)])
             for x in np.arange(xmin, xmax, GRID_SIZE) for y in np.arange(ymin, ymax, GRID_SIZE)]
    grid = gpd.GeoDataFrame({'geometry': polys}, crs="EPSG:5179")
    grid['geometry'] = grid.geometry.apply(lambda g: affinity.rotate(g, ang, origin=cen))
    grid = gpd.clip(grid, bmeter).reset_index(drop=True)
    grid = grid[grid.geometry.geom_type == 'Polygon'].reset_index(drop=True)
    grid['grid_id'] = grid.index + 1

    joined = gpd.sjoin(grid, pts, how="inner", predicate="intersects")
    stats = joined.groupby('grid_id')[['OM']].mean().reset_index()
    grid = grid.merge(stats, on='grid_id', how='left')
    if grid.crs is None:
        grid.set_crs("EPSG:5179", inplace=True)
    grid = M.fill_nan_by_idw(grid, ['OM'], k=5, power=2)
    grid['OM'] = grid['OM'].fillna(grid['OM'].mean())

    # 순수 N 요구량 (비료 N=100% 가정으로 F_Need_10a == N_Basal_10a 얻기)
    calc = FertilizerCalculator(grid, crop_type=CROP, target_yield=TARGET_YIELD,
                                basal_ratio=BASAL_RATIO, fertilizer_n_content=1.0,
                                soil_texture='식양질', min_n_limit=MIN_N)
    grid = calc.execute()
    grid['N_Basal_10a'] = grid['N_Basal_10a']          # 순수 질소 kg/10a
    grid['area_ha'] = grid.geometry.area / 10000.0
    return grid, boundary_geom, ang


# ===== 2. 비료별 처방 + PNG =====
def make_map(grid_wgs, fert, out_path, fixed_vmax=None):
    fid, maker, name, N, P, K, group, tier = fert
    rate = grid_wgs['N_Basal_10a'] / (N / 100.0) * 10.0    # kg/ha (연속)

    # 5등급
    if rate.nunique() > NUM_CLASSES:
        _, bins = pd.cut(rate, bins=NUM_CLASSES, retbins=True, duplicates='drop')
        mids = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        dose = pd.cut(rate, bins=bins, labels=mids, include_lowest=True).astype(float)
        cls = pd.cut(rate, bins=bins, labels=range(len(mids)), include_lowest=True).astype(int)
        nclass = len(mids)
    else:
        dose = rate.round(1)
        cls = dose.rank(method='dense').astype(int) - 1
        nclass = int(cls.max()) + 1

    g = grid_wgs.copy()
    g['dose'] = dose.round().astype(int)
    g['cls'] = cls

    total_kg = float((g['dose'] * g['area_ha']).sum())
    bags = total_kg / BAG_KG
    avg = total_kg / g['area_ha'].sum()
    rmin, rmax = int(g['dose'].min()), int(g['dose'].max())
    ok_max = rmax <= MACHINE_MAX_KGHA
    ok_min = rmin >= MACHINE_MIN_KGHA
    verdict = "적합" if (ok_max and ok_min) else ("최대초과" if not ok_max else "최소미달")

    # --- 그리기 ---
    fig, ax = plt.subplots(figsize=(8.2, 8.6), dpi=130)
    cmap = ListedColormap(CLASS_COLORS[:nclass])
    for c in range(nclass):
        sub = g[g['cls'] == c]
        if len(sub):
            sub.plot(ax=ax, color=CLASS_COLORS[c], edgecolor='white', linewidth=0.3)
    ax.set_axis_off()
    ax.set_aspect('equal')

    # 등급 범례 (kg/ha + 셀수 + 면적)
    handles = []
    for c in range(nclass):
        sub = g[g['cls'] == c]
        if len(sub) == 0:
            continue
        rv = int(sub['dose'].iloc[0])
        handles.append(Patch(facecolor=CLASS_COLORS[c],
                             label=f"{c+1}등급  {rv:>4} kg/ha  ({len(sub)}셀·{sub['area_ha'].sum():.2f}ha)"))
    ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.9,
              title="등급별 살포량", title_fontsize=10)

    title = f"[{fid}] {maker} {name}  ({N}-{P}-{K})"
    ax.set_title(title, fontsize=15, weight='bold', pad=12)

    mark = "◀ PM 검토대상" if fid == "F04" else ("올코팅 1차" if tier == 1 else "N코팅 2차")
    info = (f"■ 총 필요량  {total_kg:,.0f} kg  ({bags:.1f}포 · 20kg)\n"
            f"■ 평균 살포량  {avg:,.0f} kg/ha    범위 {rmin}~{rmax} kg/ha\n"
            f"■ N 함량 {N}%  →  질소 1kg당 비료 {100/N:.2f}kg 필요\n"
            f"■ 기계범위(120~900) 판정: {verdict}    [{mark}]")
    ax.text(0.5, -0.06, info, transform=ax.transAxes, ha='center', va='top',
            fontsize=11, family='Malgun Gothic',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#f5f5f5', edgecolor='#bbb'))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()

    return dict(fid=fid, maker=maker, name=name, N=N, P=P, K=K, group=group, tier=tier,
                total_kg=total_kg, bags=bags, avg=avg, rmin=rmin, rmax=rmax, verdict=verdict)


# ===== 3. 총량 비교 바차트 =====
def make_summary(rows, out_path):
    rows = sorted(rows, key=lambda r: r['N'])   # N% 오름차순 → 최소함량 이슈 가시화
    labels = [f"{r['name']}\n{r['N']}%" for r in rows]
    totals = [r['total_kg'] for r in rows]
    colors = ['#d73027' if r['fid'] == 'F04' else ('#4575b4' if r['tier'] == 1 else '#91bfdb')
              for r in rows]

    fig, ax = plt.subplots(figsize=(16, 8), dpi=130)
    bars = ax.bar(range(len(rows)), totals, color=colors, edgecolor='white')
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_ylabel("필지 총 필요량 (kg)", fontsize=12)
    ax.set_title(f"{FIELD} 벼 처방 — 비료별 총 필요량 (N 함량 오름차순)  ※ 동일 논·동일 질소 공급 기준",
                 fontsize=14, weight='bold', pad=14)
    for i, (b, r) in enumerate(zip(bars, rows)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{r['total_kg']:.0f}kg\n{r['bags']:.1f}포", ha='center', va='bottom', fontsize=7.5)
    base = min(totals)
    ax.axhline(base, color='#4575b4', ls='--', lw=0.8, alpha=0.6)
    legend = [Patch(facecolor='#4575b4', label='올코팅 1차 적용대상'),
              Patch(facecolor='#91bfdb', label='N코팅 2차 후보'),
              Patch(facecolor='#d73027', label='누보 하이코트 (PM 검토대상)')]
    ax.legend(handles=legend, fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    os.makedirs(MAP_DIR, exist_ok=True)
    print("=" * 70)
    print(f" {FIELD} 벼 변량시비 — 호산 적합 비료별 처방맵/총량")
    print("=" * 70)
    grid, boundary_geom, ang = build_base_grid()
    grid_wgs = grid.to_crs(4326)
    grid_wgs['N_Basal_10a'] = grid['N_Basal_10a'].values
    grid_wgs['area_ha'] = grid['area_ha'].values

    nb = grid['N_Basal_10a']
    print(f"\n[필지] 셀 {len(grid)}개 · 면적 {grid['area_ha'].sum():.3f}ha · 격자 {GRID_SIZE}m")
    print(f"[토양] OM {grid['OM'].min():.2f}~{grid['OM'].max():.2f} (평균 {grid['OM'].mean():.2f})"
          f"  ※ 유효규산(SI) 데이터 없음 → 공식상 기본값 150 고정")
    print(f"[벼 순수질소] {nb.min():.2f}~{nb.max():.2f} kg/10a (목표수량 {TARGET_YIELD}, 밑거름 {BASAL_RATIO}%)")
    print()

    # 올코팅 10종만 개별 지도 (1차 적용대상), 전 25종은 총량 비교에 포함
    rows = []
    for fert in FERTILIZERS:
        out = os.path.join(MAP_DIR, f"{fert[0]}_{fert[1]}_{fert[2]}.png".replace(' ', ''))
        make_individual = (fert[7] == 1)   # 올코팅 1차만 개별 PNG
        if make_individual:
            r = make_map(grid_wgs, fert, out)
        else:
            # 지도 생략, 수치만 계산
            N = fert[3]
            rate = grid_wgs['N_Basal_10a'] / (N / 100.0) * 10.0
            if rate.nunique() > NUM_CLASSES:
                _, bins = pd.cut(rate, bins=NUM_CLASSES, retbins=True, duplicates='drop')
                mids = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
                dose = pd.cut(rate, bins=bins, labels=mids, include_lowest=True).astype(float).round()
            else:
                dose = rate.round()
            tot = float((dose * grid_wgs['area_ha']).sum())
            r = dict(fid=fert[0], maker=fert[1], name=fert[2], N=N, P=fert[4], K=fert[5],
                     group=fert[6], tier=fert[7], total_kg=tot, bags=tot/BAG_KG,
                     avg=tot/grid_wgs['area_ha'].sum(), rmin=int(dose.min()), rmax=int(dose.max()),
                     verdict="적합" if dose.max() <= MACHINE_MAX_KGHA and dose.min() >= MACHINE_MIN_KGHA else "범위밖")
        rows.append(r)
        tag = "★개별지도" if make_individual else " 총량만"
        print(f"  {r['fid']} {r['maker']:>5} {r['name']:<14} N{r['N']:>2}% "
              f"| 총 {r['total_kg']:>6,.0f}kg ({r['bags']:>4.1f}포) "
              f"| 평균 {r['avg']:>3.0f} 범위 {r['rmin']}~{r['rmax']} kg/ha | {r['verdict']} {tag}")

    make_summary(rows, os.path.join(RESULT_DIR, "비료별_총량_비교.png"))

    # 요약 CSV
    pd.DataFrame(rows).to_csv(os.path.join(RESULT_DIR, "비료별_총량_요약.csv"),
                             index=False, encoding='utf-8-sig')

    print(f"\n[완료] {os.path.abspath(RESULT_DIR)}")
    print(f"  · 개별 처방맵 PNG {sum(1 for f in FERTILIZERS if f[7]==1)}장 (올코팅 1차) → maps/")
    print(f"  · 총량 비교 차트 1장 (전 25종)")
    print(f"  · 요약 CSV 1개")


if __name__ == "__main__":
    main()
