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

# [필수] Rasterio 라이브러리 (DJI Tiff 생성용)
try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.features import rasterize
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError:
    print("[경고] 'rasterio' 라이브러리가 없습니다. DJI 처방맵 생성을 위해 'pip install rasterio'를 설치해주세요.")
    rasterio = None

from fertilizer_calculator import FertilizerCalculator

# ======================================================
# 0. 환경 설정
# ======================================================
DATA_FOLDER = "test_data"
RESULT_ROOT = "result"

# 비료 처방 옵션
CROP_TYPE = 'rice'
TARGET_YIELD = 480
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

    gdf['iso_rate'] = (gdf[rate_col] * 1000).astype(int)

    zone_id = 0
    for rate_val, group in gdf.groupby('iso_rate'):
        zone_id += 1
        tzn = ET.SubElement(task, "TZN", {"TreatmentZoneCode": str(zone_id), "TreatmentZoneDesignator": f"Rate_{rate_val}"})
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

def export_csv(gdf, output_folder, base_name):
    """
    CSV 내보내기 (grid_id, F_Need_10a, F_Total, Center_Lat, Center_Lon, Vertices...)
    """
    if gdf.crs is None:
        gdf.set_crs("EPSG:5179", inplace=True)

    centroids_meter = gdf.geometry.centroid
    centroids_wgs = centroids_meter.to_crs(epsg=4326)
    gdf_wgs = gdf.to_crs(epsg=4326)

    gdf_wgs['Center_Lat'] = centroids_wgs.y
    gdf_wgs['Center_Lon'] = centroids_wgs.x

    vertex_data = []
    max_v = 0

    for idx, row in gdf_wgs.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            row_dict = {'grid_id': row['grid_id']}
            for i, (lon, lat) in enumerate(coords):
                row_dict[f'V{i+1}_Lat'] = lat
                row_dict[f'V{i+1}_Lon'] = lon
            if len(coords) > max_v: max_v = len(coords)
            vertex_data.append(row_dict)

    v_df = pd.DataFrame(vertex_data)
    out_df = gdf_wgs.drop(columns=['geometry']).merge(v_df, on='grid_id', how='left')

    base_cols = ['grid_id', 'F_Need_10a', 'F_Total', 'Center_Lat', 'Center_Lon']
    v_cols = []
    for i in range(max_v):
        v_cols.append(f'V{i+1}_Lat')
        v_cols.append(f'V{i+1}_Lon')

    final_cols = base_cols + v_cols
    final_cols = [c for c in final_cols if c in out_df.columns]

    csv_path = os.path.join(output_folder, f"{base_name}_Result.csv")
    out_df[final_cols].to_csv(csv_path, index=False, encoding='cp949')
    print(f"  - CSV 저장 완료: {os.path.basename(csv_path)}")

def export_dji_tif(gdf, output_folder, base_name, rate_col='F_Need_10a'):
    """
    [수정] DJI 드론용 처방맵 생성
    - Nodata(0) 처리를 강화하여 필지 바깥 영역을 투명하게 만듦 (Clipping 효과)
    """
    if rasterio is None: return

    # 좌표계 안전장치
    src_crs = gdf.crs
    if src_crs is None:
        print("  [경고] 데이터의 좌표계(CRS)가 유실되어 'EPSG:5179'로 강제 설정합니다.")
        gdf.set_crs("EPSG:5179", inplace=True)
        src_crs = gdf.crs

    # 1. 단위 변환
    gdf['DJI_Rate'] = gdf[rate_col] * 10

    # [검증 로그]
    min_rate = gdf['DJI_Rate'].min()
    max_rate = gdf['DJI_Rate'].max()
    mean_rate = gdf['DJI_Rate'].mean()
    print(f"  [DJI-TIF 검증] 살포량(kg/ha) 범위: 최소 {min_rate:.1f} ~ 최대 {max_rate:.1f} (평균 {mean_rate:.1f})")

    # 2. 래스터화 준비
    pixel_size = 1.0
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))

    # 좌표 변환 행렬 (미터 좌표계)
    src_transform = from_origin(minx, maxy, pixel_size, pixel_size)

    # 3. 래스터 태우기 (Burn)
    # fill=0 : 데이터가 없는 배경(필지 바깥)을 0으로 채움
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['DJI_Rate']))
    src_array = rasterize(
        shapes,
        out_shape=(height, width),
        transform=src_transform,
        fill=0,
        default_value=0,
        dtype='float32',
        all_touched=True
    )

    # 4. 좌표계 변환 (EPSG:5179 -> EPSG:4326 WGS84)
    dst_crs = 'EPSG:4326'

    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left=minx, bottom=miny, right=maxx, top=maxy
    )

    dst_array = np.zeros((dst_height, dst_width), dtype='float32')

    # [핵심 수정] src_nodata=0, dst_nodata=0 설정
    # 변환 과정에서 0인 값(배경)을 '데이터 없음'으로 확실하게 처리함
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,        # 원본의 0은 데이터 없음임
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,        # 결과의 0도 데이터 없음으로 처리
        resampling=Resampling.nearest
    )

    # 5. GeoTIFF 저장
    tif_path = os.path.join(output_folder, f"{base_name}_DJI.tif")
    with rasterio.open(
        tif_path, 'w', driver='GTiff',
        height=dst_height, width=dst_width,
        count=1, dtype='float32',
        crs=dst_crs, transform=dst_transform,
        nodata=0  # [핵심] 파일 헤더에 '0은 투명한 값'이라고 명시
    ) as dst:
        dst.write(dst_array, 1)

    # 6. TFW 파일 생성
    tfw_path = os.path.join(output_folder, f"{base_name}_DJI.tfw")
    with open(tfw_path, 'w') as f:
        f.write(f"{dst_transform.a}\n")
        f.write(f"{dst_transform.b}\n")
        f.write(f"{dst_transform.d}\n")
        f.write(f"{dst_transform.e}\n")
        f.write(f"{dst_transform.c + dst_transform.a / 2}\n")
        f.write(f"{dst_transform.f + dst_transform.e / 2}\n")

    print(f"  - DJI 처방맵(TIF, TFW) 저장 완료: {os.path.basename(tif_path)}")


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
        soil_input = f"zip://{soil_path}".replace("\\", "/") if not soil_path.startswith("zip://") else soil_path.replace("\\", "/")
        bound_input = f"zip://{boundary_path}".replace("\\", "/") if not boundary_path.startswith("zip://") else boundary_path.replace("\\", "/")

        points_gdf = None
        encodings = ['euc-kr', 'cp949', 'latin1', 'utf-8']
        for enc in encodings:
            try:
                points_gdf = gpd.read_file(soil_input, encoding=enc)
                break
            except: continue
        if points_gdf is None: points_gdf = gpd.read_file(soil_input)
        boundary_gdf = gpd.read_file(bound_input)

        points_meter = fix_coordinate_system(points_gdf, "토양점")
        boundary_meter = fix_coordinate_system(boundary_gdf, "경계")
        if points_meter is None or boundary_meter is None: return

        # 데이터 정제
        new_columns = {}
        for col in points_meter.columns:
            clean_col = re.sub(r'[^A-Z0-9]', '', col.upper())
            new_columns[col] = clean_col
        points_meter = points_meter.rename(columns=new_columns)
        points_meter = points_meter.loc[:, ~points_meter.columns.duplicated()]

        if 'GEOMETRY' in points_meter.columns:
            points_meter = points_meter.rename(columns={'GEOMETRY': 'geometry'})
        points_meter = points_meter.set_geometry('geometry')

        exclude_cols = ['ID', 'LATITUDE', 'LONGITUDE', 'COUNTRATE', 'geometry']
        for col in points_meter.columns:
            if col not in exclude_cols:
                points_meter[col] = pd.to_numeric(points_meter[col], errors='coerce')
                if points_meter[col].isnull().sum() > 0:
                     points_meter[col] = points_meter[col].astype(str).str.extract(r'([-+]?\d*\.?\d+)')[0]
                     points_meter[col] = pd.to_numeric(points_meter[col], errors='coerce')
                points_meter[col] = points_meter[col].fillna(0)

        numeric_cols = points_meter.select_dtypes(include=[np.number]).columns.tolist()
        analysis_cols = [col for col in numeric_cols if col not in exclude_cols]
        for ess in ['OM', 'SI', 'PH', 'P', 'K', 'MG']:
            if ess in points_meter.columns and ess not in analysis_cols: analysis_cols.append(ess)

        print(f"  - 분석 대상 성분: {analysis_cols}")
        if 'OM' in points_meter.columns:
            print(f"  [데이터검증] 로드된 유기물(OM) 평균: {points_meter['OM'].mean():.2f}")

    except Exception as e:
        print(f"[오류] 전처리 실패: {e}")
        return

    boundary_geom = boundary_meter.union_all()
    rotation_angle = get_main_angle(boundary_geom)
    centroid = boundary_geom.centroid
    rotated_boundary = affinity.rotate(boundary_geom, -rotation_angle, origin=centroid)
    xmin, ymin, xmax, ymax = rotated_boundary.bounds
    grid_size = 5
    cols = np.arange(xmin, xmax, grid_size)
    rows = np.arange(ymin, ymax, grid_size)
    polygons = [Polygon([(x, y), (x+grid_size, y), (x+grid_size, y+grid_size), (x, y+grid_size)]) for x in cols for y in rows]
    temp_grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:5179")
    temp_grid['geometry'] = temp_grid['geometry'].apply(lambda g: affinity.rotate(g, rotation_angle, origin=centroid))
    clipped_grid = gpd.clip(temp_grid, boundary_meter).reset_index(drop=True)

    # Grid ID 통일
    clipped_grid['grid_id'] = (clipped_grid.index + 1).astype(int)

    joined = gpd.sjoin(clipped_grid, points_meter, how="inner", predicate="intersects")
    print(f"  - 공간 조인 결과: 총 {len(joined)}개의 토양 점이 매칭되었습니다.")

    grid_stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()
    grid_stats['grid_id'] = grid_stats['grid_id'].astype(int)

    final_grid = clipped_grid.merge(grid_stats, on='grid_id', how='left')

    # [중요] Merge 후 CRS 유실 방지
    if final_grid.crs is None:
        final_grid.set_crs(clipped_grid.crs, inplace=True)

    final_grid[analysis_cols] = final_grid[analysis_cols].fillna(0)

    if 'OM' in final_grid.columns:
        valid_om = final_grid[final_grid['OM'] > 0]['OM']
        print(f"  [최종검증] 계산 직전 유기물(OM) > 0 인 그리드 개수: {len(valid_om)}")
        if len(valid_om) > 0:
            print(f"  [최종검증] 해당 그리드들의 평균 OM: {valid_om.mean():.2f}")

    # 계산기 실행
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

    # 1. SHP 저장
    output_shp = os.path.join(field_result_dir, f"{base_name}_Result.shp")
    final_grid_renamed[actual_save_cols].to_file(output_shp, encoding='euc-kr')
    print(f"  - SHP 저장 완료: {os.path.basename(output_shp)}")

    # 2. CSV 저장
    export_csv(final_grid_renamed, field_result_dir, base_name)

    # 3. DJI Tiff 저장
    export_dji_tif(final_grid_renamed, field_result_dir, base_name, rate_col='F_Need_10a')

    # 4. ISOXML 저장
    try:
        isoxml_gdf = final_grid.to_crs(epsg=4326)
        export_isoxml(isoxml_gdf, field_result_dir, task_name=base_name, rate_col='F_Need_10a')
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
    if not soil_files: print(f"[알림] 파일이 없습니다."); return

    print(f"==========================================")
    print(f" 설정: {CROP_TYPE}, 토성: {SOIL_TEXTURE}")
    print(f" 최소시비량: {MIN_N_REQUIREMENT}kg/10a")
    print(f"==========================================")

    for soil_file in soil_files:
        matched_boundary, base_name = find_matching_boundary(soil_file, boundary_files)
        if matched_boundary: process_single_field(soil_file, matched_boundary, base_name)
        else: print(f"\n[건너뜀] 매칭 실패: {os.path.basename(soil_file)}")
    print(f"\n[완료] 결과 폴더: {os.path.abspath(RESULT_ROOT)}")

if __name__ == "__main__":
    main()