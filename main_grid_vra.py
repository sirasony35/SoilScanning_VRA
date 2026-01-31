import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity
import math
import os
import xml.etree.ElementTree as ET

# [중요] 분리한 모듈 불러오기
# fertilizer_calculator.py 파일이 같은 폴더에 있어야 합니다.
from fertilizer_calculator import FertilizerCalculator

# ======================================================
# 1. 파일 경로 설정
# ======================================================
folder_name = "test_data"
file_name = "Daedong_Daedong_AM_Buan_gun_0.55Ha_Shapefile.zip"
boundary_file_name = "Wondermilk_Daedong_Daedong_AM-Buan-gun0.55Ha_Boundary.zip"

full_path = os.path.abspath(os.path.join(folder_name, file_name))
boundary_full_path = os.path.abspath(os.path.join(folder_name, boundary_file_name))

point_zip_path = f"zip://{full_path}".replace("\\", "/")
boundary_zip_path = f"zip://{boundary_full_path}".replace("\\", "/")

# ======================================================
# 2. 데이터 로드 및 전처리
# ======================================================
print(">>> 데이터 로딩 및 좌표계 변환...")
try:
    points_gdf = gpd.read_file(point_zip_path)
    boundary_gdf = gpd.read_file(boundary_zip_path)

    if points_gdf.crs is None: points_gdf.crs = "EPSG:4326"
    if boundary_gdf.crs is None: boundary_gdf.crs = "EPSG:4326"

    points_meter = points_gdf.to_crs(epsg=5179)
    boundary_meter = boundary_gdf.to_crs(epsg=5179)
except Exception as e:
    print(f"파일 로드 오류: {e}")
    exit()

# 자동 컬럼 식별
exclude_cols = ['Id', 'Latitude', 'Longitude', 'Countrate', 'geometry']
numeric_cols = points_meter.select_dtypes(include=[np.number]).columns.tolist()
analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
print(f"분석 성분: {analysis_cols}")


# ======================================================
# 3. 그리드 생성 (회전 적용)
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


print(">>> 필지 방향 분석 및 그리드 생성...")
boundary_geom = boundary_meter.union_all()
rotation_angle = get_main_angle(boundary_geom)
centroid = boundary_geom.centroid

# 회전 -> 그리드 -> 역회전 -> Clip
rotated_boundary = affinity.rotate(boundary_geom, -rotation_angle, origin=centroid)
xmin, ymin, xmax, ymax = rotated_boundary.bounds
grid_size = 5

cols = np.arange(xmin, xmax, grid_size)
rows = np.arange(ymin, ymax, grid_size)
polygons = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)])
            for x in cols for y in rows]

temp_grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:5179")
temp_grid['geometry'] = temp_grid['geometry'].apply(lambda g: affinity.rotate(g, rotation_angle, origin=centroid))

clipped_grid = gpd.clip(temp_grid, boundary_meter).reset_index(drop=True)
clipped_grid['grid_id'] = clipped_grid.index + 1

# ======================================================
# 4. 공간 조인 (Spatial Join)
# ======================================================
joined = gpd.sjoin(clipped_grid, points_meter, how="inner", predicate="intersects")
grid_stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()

final_grid = clipped_grid.merge(grid_stats, on='grid_id', how='left')
final_grid[analysis_cols] = final_grid[analysis_cols].fillna(0)

# ======================================================
# 5. [모듈 사용] 비료량 산출 (Fertilizer Calculator)
# ======================================================
print(">>> 비료량 산출 모듈 실행...")

# Calculator 클래스 인스턴스 생성 및 실행
calculator = FertilizerCalculator(final_grid)
final_grid = calculator.execute()

# ======================================================
# 6. 결과 저장 (SHP & ISOXML)
# ======================================================
result_dir = "result"
os.makedirs(result_dir, exist_ok=True)

# (1) SHP 저장
output_shp = os.path.join(result_dir, "Result_Final_VRA_Map.shp")

# N_Need_area -> N_Total (총량이라는 뜻으로 변경)
final_grid_renamed = final_grid.rename(columns={'N_Need_area': 'N_Total'})
# 변경된 이름으로 저장 리스트 구성
save_cols = ['grid_id', 'Grid_Area', 'N_Need_10a', 'N_Total', 'geometry'] + analysis_cols
# 저장
final_grid_renamed[save_cols].to_file(output_shp, encoding='euc-kr')
print(f"[SHP 저장] {os.path.abspath(output_shp)}")
s
# (2) ISOXML 저장 (함수 정의)
def export_isoxml(gdf, output_folder):
    taskdata_dir = os.path.join(output_folder, "TASKDATA")
    os.makedirs(taskdata_dir, exist_ok=True)

    root = ET.Element("ISO11783_TaskData", {"VersionMajor": "4", "VersionMinor": "3", "DataTransferOrigin": "1"})
    task = ET.SubElement(root, "TSK", {"TaskDesignator": "VRA_Task", "TaskStatus": "1"})

    # mg/m2 단위 변환 (Rate)
    gdf['iso_rate'] = (gdf['N_Need_10a'] * 1000).astype(int)

    zone_id = 0
    for rate_val, group in gdf.groupby('iso_rate'):
        zone_id += 1
        tzn = ET.SubElement(task, "TZN",
                            {"TreatmentZoneCode": str(zone_id), "TreatmentZoneDesignator": f"Rate_{rate_val}"})
        ET.SubElement(tzn, "PDV", {"ProcessDataDDI": "0006", "ProcessDataValue": str(rate_val)})

        for _, row in group.iterrows():
            ptn = ET.SubElement(tzn, "PTN")
            lsg = ET.SubElement(ptn, "LSG", {"LineStringType": "1"})
            for lon, lat in zip(*row['geometry'].exterior.coords.xy):
                ET.SubElement(lsg, "PNT", {"A": f"{lat:.9f}", "B": f"{lon:.9f}"})

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(os.path.join(taskdata_dir, "TASKDATA.XML"), encoding="UTF-8", xml_declaration=True)
    return os.path.join(taskdata_dir, "TASKDATA.XML")


# ISOXML 변환 실행 (위경도 변환 후)
print(">>> ISOXML 변환 및 저장...")
try:
    isoxml_gdf = final_grid.to_crs(epsg=4326)
    xml_path = export_isoxml(isoxml_gdf, result_dir)
    print(f"[ISOXML 저장] {os.path.abspath(xml_path)}")
except Exception as e:
    print(f"ISOXML 오류: {e}")

print("\n[완료] 모든 작업이 끝났습니다.")