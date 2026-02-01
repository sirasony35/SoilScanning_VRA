import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely import affinity
import math
import os
import glob
import xml.etree.ElementTree as ET
import re

from fertilizer_calculator import FertilizerCalculator

# ======================================================
# 0. 환경 설정
# ======================================================
DATA_FOLDER = "test_data"
RESULT_ROOT = "result"

# 비료 처방 옵션
CROP_TYPE = 'soybean'
TARGET_YIELD = 500
BASAL_RATIO = 100
SOIL_TEXTURE = '식양질'
MIN_N_REQUIREMENT = 2.0

# 비료 제품 정보
FERTILIZER_N_CONTENT = 0.20
FERTILIZER_BAG_WEIGHT = 20


# ======================================================
# 1. 공통 함수 정의
# ======================================================

def get_main_angle(geometry):
    rect = geometry.minimum_rotated_rectangle
    coords = list(rect.exterior.coords)
    max_len = 0
    main_angle = 0
    for i in range(len(coords) - 1):
        dx = coords[i + 1][0] - coords[i][0]
        dy = coords[i + 1][1] - coords[i][1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length > max_len:
            max_len = length
            main_angle = math.degrees(math.atan2(dy, dx))
    return main_angle


def export_isoxml(gdf, output_folder, task_name, rate_col='F_Need_10a'):
    taskdata_dir = os.path.join(output_folder, "TASKDATA")
    os.makedirs(taskdata_dir, exist_ok=True)

    root = ET.Element("ISO11783_TaskData", {"VersionMajor": "4", "VersionMinor": "3", "DataTransferOrigin": "1"})
    task = ET.SubElement(root, "TSK", {"TaskDesignator": task_name, "TaskStatus": "1"})

    # mg/m2 단위 변환
    gdf['iso_rate'] = (gdf[rate_col] * 1000).astype(int)

    zone_id = 0
    for rate_val, group in gdf.groupby('iso_rate'):
        zone_id += 1
        tzn = ET.SubElement(task, "TZN",
                            {"TreatmentZoneCode": str(zone_id), "TreatmentZoneDesignator": f"Rate_{rate_val}"})
        ET.SubElement(tzn, "PDV", {"ProcessDataDDI": "0006", "ProcessDataValue": str(rate_val)})

        for _, row in group.iterrows():
            if row['geometry'].geom_type != 'Polygon': continue
            ptn = ET.SubElement(tzn, "PTN")
            lsg = ET.SubElement(ptn, "LSG", {"LineStringType": "1"})
            for lon, lat in zip(*row['geometry'].exterior.coords.xy):
                ET.SubElement(lsg, "PNT", {"A": f"{lat:.9f}", "B": f"{lon:.9f}"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    xml_path = os.path.join(taskdata_dir, "TASKDATA.XML")
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
    return xml_path


def fix_coordinate_system(gdf, tag):
    if gdf.crs is None:
        print(f"  [보정] '{tag}' 좌표계가 정의되지 않아 EPSG:4326(위경도)로 설정합니다.")
        gdf.crs = "EPSG:4326"

    bounds = gdf.total_bounds
    mean_x = (bounds[0] + bounds[2]) / 2
    mean_y = (bounds[1] + bounds[3]) / 2

    if (30 < mean_x < 45) and (120 < mean_y < 135):
        print(f"  [!!! 좌표보정 !!!] '{tag}' 데이터의 위도/경도가 반대로 되어있습니다. (Swap 실행)")
        gdf['geometry'] = gdf['geometry'].apply(lambda p: Point(p.y, p.x) if p.geom_type == 'Point' else p)

    try:
        gdf_meter = gdf.to_crs(epsg=5179)
        return gdf_meter
    except Exception as e:
        print(f"  [오류] 좌표 변환 실패: {e}")
        return None


def process_single_field(soil_path, boundary_path, base_name):
    print(f"\n>>> [처리 시작] {base_name}")
    try:
        soil_input = f"zip://{soil_path}".replace("\\", "/") if not soil_path.startswith(
            "zip://") else soil_path.replace("\\", "/")
        bound_input = f"zip://{boundary_path}".replace("\\", "/") if not boundary_path.startswith(
            "zip://") else boundary_path.replace("\\", "/")

        # 파일 읽기
        points_gdf = None
        encodings_to_try = ['euc-kr', 'cp949', 'latin1', 'utf-8']
        for enc in encodings_to_try:
            try:
                points_gdf = gpd.read_file(soil_input, encoding=enc)
                break
            except:
                continue
        if points_gdf is None: points_gdf = gpd.read_file(soil_input)
        boundary_gdf = gpd.read_file(bound_input)

        # 좌표계 보정
        points_meter = fix_coordinate_system(points_gdf, "토양점")
        boundary_meter = fix_coordinate_system(boundary_gdf, "경계")

        if points_meter is None or boundary_meter is None: return

        # ------------------------------------------------------------------
        # [데이터 정제 강화 (Regex 복구 및 중복 제거)
        # ------------------------------------------------------------------
        # 1. 컬럼명 정제
        new_columns = {}
        for col in points_meter.columns:
            clean_col = re.sub(r'[^A-Z0-9]', '', col.upper())
            new_columns[col] = clean_col
        points_meter = points_meter.rename(columns=new_columns)

        # 중복 컬럼 제거 (같은 이름의 컬럼이 여러 개면 계산 오류 발생)
        points_meter = points_meter.loc[:, ~points_meter.columns.duplicated()]

        # 2. Geometry 복구
        if 'GEOMETRY' in points_meter.columns:
            points_meter = points_meter.rename(columns={'GEOMETRY': 'geometry'})
        points_meter = points_meter.set_geometry('geometry')

        exclude_cols = ['ID', 'LATITUDE', 'LONGITUDE', 'COUNTRATE', 'geometry']

        # 3. 숫자 강제 추출 (지난번 pass 버그 수정!)
        for col in points_meter.columns:
            if col not in exclude_cols:
                # 1차 시도: 숫자 변환
                points_meter[col] = pd.to_numeric(points_meter[col], errors='coerce')

                # 2차 시도: NaN이 있다면 Regex로 강제 추출 (데이터가 더러울 경우 대비)
                if points_meter[col].isnull().sum() > 0:
                    points_meter[col] = points_meter[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)')
                    points_meter[col] = pd.to_numeric(points_meter[col], errors='coerce')

                # 최종: 그래도 없으면 0
                points_meter[col] = points_meter[col].fillna(0)

        numeric_cols = points_meter.select_dtypes(include=[np.number]).columns.tolist()
        analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

        for essential in ['OM', 'SI', 'PH', 'P', 'K', 'MG']:
            if essential in points_meter.columns and essential not in analysis_cols:
                analysis_cols.append(essential)

        print(f"  - 분석 대상 성분: {analysis_cols}")

        if 'OM' in points_meter.columns:
            print(f"  [데이터검증] 로드된 유기물(OM) 평균: {points_meter['OM'].mean():.2f}")

    except Exception as e:
        print(f"[오류] 파일 로드/전처리 실패: {e}")
        return

    # 그리드 생성
    boundary_geom = boundary_meter.union_all()
    rotation_angle = get_main_angle(boundary_geom)
    centroid = boundary_geom.centroid

    rotated_boundary = affinity.rotate(boundary_geom, -rotation_angle, origin=centroid)
    xmin, ymin, xmax, ymax = rotated_boundary.bounds
    grid_size = 5

    cols = np.arange(xmin, xmax, grid_size)
    rows = np.arange(ymin, ymax, grid_size)
    polygons = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) for x in cols
                for y in rows]

    temp_grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:5179")
    temp_grid['geometry'] = temp_grid['geometry'].apply(lambda g: affinity.rotate(g, rotation_angle, origin=centroid))

    clipped_grid = gpd.clip(temp_grid, boundary_meter).reset_index(drop=True)
    clipped_grid['grid_id'] = clipped_grid.index + 1

    # [수정] grid_id 타입 강제 통일 (중요!)
    clipped_grid['grid_id'] = clipped_grid['grid_id'].astype(int)

    # 공간 조인
    joined = gpd.sjoin(clipped_grid, points_meter, how="inner", predicate="intersects")
    print(f"  - 공간 조인 결과: 총 {len(joined)}개의 토양 점이 매칭되었습니다.")

    # 그리드별 평균 계산
    grid_stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()
    grid_stats['grid_id'] = grid_stats['grid_id'].astype(int)  # 타입 통일

    # 그리드에 데이터 병합
    final_grid = clipped_grid.merge(grid_stats, on='grid_id', how='left')
    final_grid[analysis_cols] = final_grid[analysis_cols].fillna(0)

    # [최종 검증] 계산기로 넘어가기 직전의 데이터 확인
    if 'OM' in final_grid.columns:
        final_om_mean = final_grid[final_grid['OM'] > 0]['OM'].mean()
        print(f"  [최종검증] 계산 직전 그리드 유기물(OM) 평균: {final_om_mean:.2f} (0 제외)")
        if pd.isna(final_om_mean) or final_om_mean == 0:
            print("  [!!! 치명적 오류 !!!] 데이터 병합(Merge) 과정에서 유기물 데이터가 유실되었습니다. grid_id 매칭 실패 가능성.")

    # 비료량 산출
    calculator = FertilizerCalculator(
        final_grid,
        crop_type=CROP_TYPE,
        target_yield=TARGET_YIELD,
        basal_ratio=BASAL_RATIO,
        fertilizer_n_content=FERTILIZER_N_CONTENT,
        soil_texture=SOIL_TEXTURE,
        min_n_limit=MIN_N_REQUIREMENT
    )
    final_grid = calculator.execute()

    # 결과 저장
    field_result_dir = os.path.join(RESULT_ROOT, base_name)
    os.makedirs(field_result_dir, exist_ok=True)

    final_grid_renamed = final_grid.rename(columns={
        'N_Need_10a': 'N_Need_10a',
        'N_Total': 'N_Total',
        'F_Need_10a': 'F_Need_10a',
        'F_Total': 'F_Total'
    })

    save_cols = ['grid_id', 'Grid_Area', 'N_Need_10a', 'N_Total', 'F_Need_10a', 'F_Total', 'geometry'] + analysis_cols
    actual_save_cols = [c for c in save_cols if c in final_grid_renamed.columns]

    output_shp = os.path.join(field_result_dir, f"{base_name}_Result.shp")
    final_grid_renamed[actual_save_cols].to_file(output_shp, encoding='euc-kr')
    print(f"  - SHP 저장 완료: {os.path.basename(output_shp)}")

    try:
        isoxml_gdf = final_grid.to_crs(epsg=4326)
        xml_path = export_isoxml(isoxml_gdf, field_result_dir, task_name=base_name, rate_col='F_Need_10a')
        print(f"  - ISOXML 저장 완료")
    except Exception as e:
        print(f"  - ISOXML 생성 오류: {e}")


def find_matching_boundary(soil_file, all_boundary_files):
    folder, filename = os.path.split(soil_file)
    core_name = filename.replace("_Shapefile.zip", "")
    exact_match = os.path.join(folder, f"{core_name}_Boundary.zip")
    if exact_match in all_boundary_files: return exact_match, core_name
    candidates = [b for b in all_boundary_files if core_name in os.path.basename(b)]
    if len(candidates) == 1: return candidates[0], core_name
    if len(all_boundary_files) == 1: return all_boundary_files[0], core_name
    return None, core_name


def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"[오류] '{DATA_FOLDER}' 폴더가 없습니다.")
        return

    soil_files = glob.glob(os.path.join(DATA_FOLDER, "*_Shapefile.zip"))
    boundary_files = glob.glob(os.path.join(DATA_FOLDER, "*_Boundary.zip"))

    if not soil_files:
        print(f"[알림] 처리할 파일이 없습니다.")
        return

    print(f"==========================================")
    print(f" 설정: {CROP_TYPE}, 토성: {SOIL_TEXTURE}")
    print(f" 최소시비량: {MIN_N_REQUIREMENT}kg/10a")
    print(f"==========================================")

    for soil_file in soil_files:
        matched_boundary, base_name = find_matching_boundary(soil_file, boundary_files)
        if matched_boundary:
            process_single_field(soil_file, matched_boundary, base_name)
        else:
            print(f"\n[건너뜀] 바운더리 매칭 실패: {os.path.basename(soil_file)}")

    print(f"\n[완료] 결과 폴더: {os.path.abspath(RESULT_ROOT)}")


if __name__ == "__main__":
    main()