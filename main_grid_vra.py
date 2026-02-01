import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity
import math
import os
import glob
import xml.etree.ElementTree as ET

from fertilizer_calculator import FertilizerCalculator

# ======================================================
# 0. 환경 설정 (사용자 입력)
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


def process_single_field(soil_path, boundary_path, base_name):
    print(f"\n>>> [처리 시작] {base_name}")
    try:
        soil_input = f"zip://{soil_path}".replace("\\", "/") if not soil_path.startswith(
            "zip://") else soil_path.replace("\\", "/")
        bound_input = f"zip://{boundary_path}".replace("\\", "/") if not boundary_path.startswith(
            "zip://") else boundary_path.replace("\\", "/")

        points_gdf = gpd.read_file(soil_input)
        boundary_gdf = gpd.read_file(bound_input)

        if points_gdf.crs is None: points_gdf.crs = "EPSG:4326"
        if boundary_gdf.crs is None: boundary_gdf.crs = "EPSG:4326"

        points_meter = points_gdf.to_crs(epsg=5179)
        boundary_meter = boundary_gdf.to_crs(epsg=5179)

        # ------------------------------------------------------------------
        # [수정] 데이터 전처리 (Ghost Text 제거 및 숫자 강제 변환)
        # ------------------------------------------------------------------
        # 1. 컬럼명 대문자화 및 공백 제거
        points_meter.columns = points_meter.columns.str.strip().str.upper()

        # [핵심 수정] GEOMETRY 컬럼이 생겼다면 다시 geometry(소문자)로 복구하고 활성화
        if 'GEOMETRY' in points_meter.columns:
            points_meter = points_meter.rename(columns={'GEOMETRY': 'geometry'})

        points_meter = points_meter.set_geometry('geometry')  # 명시적 활성화

        # 2. 제외할 컬럼 정의 ('geometry'는 소문자로 확인)
        exclude_cols = ['ID', 'LATITUDE', 'LONGITUDE', 'COUNTRATE', 'geometry']

        # 3. 숫자 변환
        for col in points_meter.columns:
            if col not in exclude_cols:
                points_meter[col] = pd.to_numeric(points_meter[col], errors='coerce').fillna(0)

        # 4. 분석 대상 컬럼 확정
        numeric_cols = points_meter.select_dtypes(include=[np.number]).columns.tolist()
        analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

        # [안전장치] 필수 성분 강제 추가
        for essential in ['OM', 'SI', 'PH', 'P', 'K', 'MG']:
            if essential in points_meter.columns and essential not in analysis_cols:
                analysis_cols.append(essential)

        print(f"  - 분석 대상 성분: {analysis_cols}")

    except Exception as e:
        print(f"[오류] 파일 로드 실패: {e}")
        # import traceback; traceback.print_exc() # 디버깅용
        return

    # ------------------------------------------------------------------
    # 그리드 생성 및 분석 로직
    # ------------------------------------------------------------------
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

    # 공간 조인
    joined = gpd.sjoin(clipped_grid, points_meter, how="inner", predicate="intersects")
    grid_stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()

    final_grid = clipped_grid.merge(grid_stats, on='grid_id', how='left')
    final_grid[analysis_cols] = final_grid[analysis_cols].fillna(0)

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

    final_grid_renamed = final_grid.rename(columns={'F_Need_10a': 'F_RATE', 'F_Total': 'F_KG', 'N_Need_10a': 'N_RATE'})
    save_cols = ['grid_id', 'Grid_Area', 'N_RATE', 'F_RATE', 'F_KG', 'geometry'] + analysis_cols
    actual_save_cols = [c for c in save_cols if c in final_grid_renamed.columns]

    output_shp = os.path.join(field_result_dir, f"{base_name}_Result.shp")
    final_grid_renamed[actual_save_cols].to_file(output_shp, encoding='euc-kr')
    print(f"  - SHP 저장 완료")

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