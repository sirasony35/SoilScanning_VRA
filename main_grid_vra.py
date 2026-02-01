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
CROP_TYPE = 'soybean'  # 'rice', 'soybean', 'wheat'
TARGET_YIELD = 500  # 목표 수량 (kg/10a), 벼에만 해당
BASAL_RATIO = 100  # 밑거름 비율 (%)
SOIL_TEXTURE = '식양질'  # 토성: '사질', '사양질', '식양질' 등

#최소 질소 시비량 설정 (kg/10a)
# 계산 결과가 0 또는 이 값보다 작게 나오면 강제로 이 값으로 설정됨 (기계 작동용)
MIN_N_REQUIREMENT = 2.0

# 비료 제품 정보
FERTILIZER_N_CONTENT = 0.20  # 비료 1포대의 질소 함량
FERTILIZER_BAG_WEIGHT = 20   # 비료 1포대의 무게(kg)


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
    except Exception as e:
        print(f"[오류] 파일 로드 실패: {e}")
        return

    exclude_cols = ['Id', 'Latitude', 'Longitude', 'Countrate', 'geometry']
    numeric_cols = points_meter.select_dtypes(include=[np.number]).columns.tolist()
    analysis_cols = [col for col in numeric_cols if col not in exclude_cols]

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

    joined = gpd.sjoin(clipped_grid, points_meter, how="inner", predicate="intersects")
    grid_stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()

    final_grid = clipped_grid.merge(grid_stats, on='grid_id', how='left')
    final_grid[analysis_cols] = final_grid[analysis_cols].fillna(0)

    # ------------------------------------------------------
    # 5. 비료량 산출
    # ------------------------------------------------------
    calculator = FertilizerCalculator(
        final_grid,
        crop_type=CROP_TYPE,
        target_yield=TARGET_YIELD,
        basal_ratio=BASAL_RATIO,
        fertilizer_n_content=FERTILIZER_N_CONTENT,
        soil_texture=SOIL_TEXTURE,
        min_n_limit=MIN_N_REQUIREMENT  # [신규] 최소 시비량 전달
    )
    final_grid = calculator.execute()

    # 6. 결과 저장
    field_result_dir = os.path.join(RESULT_ROOT, base_name)
    os.makedirs(field_result_dir, exist_ok=True)

    save_cols = ['grid_id', 'Grid_Area', 'N_Need_10a', 'N_Total', 'F_Need_10a', 'F_Total', 'geometry'] + analysis_cols

    output_shp = os.path.join(field_result_dir, f"{base_name}_Result.shp")
    final_grid[save_cols].to_file(output_shp, encoding='euc-kr')
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