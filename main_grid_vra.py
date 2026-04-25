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
import datetime
import zipfile
import random

# [필수] Rasterio 라이브러리 (DJI Tiff 및 ISOXML BIN 생성용)
try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.features import rasterize
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except ImportError:
    print("[경고] 'rasterio' 라이브러리가 없습니다. 처방맵 생성을 위해 'pip install rasterio'를 설치해주세요.")
    rasterio = None

from fertilizer_calculator import FertilizerCalculator

# ======================================================
# 0. 환경 설정
# ======================================================
DATA_FOLDER = "new_data"
RESULT_ROOT = "result"

CROP_TYPE = 'soybean'
TARGET_YIELD = 500
BASAL_RATIO = 100
SOIL_TEXTURE = '식양질'
MIN_N_REQUIREMENT = 2.0

FERTILIZER_N_CONTENT = 0.20
FERTILIZER_BAG_WEIGHT = 20

GRID_SIZES = [10, 16, 20]


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


def export_isoxml(gdf, boundary_geom, output_folder, task_name, rate_col='DOSE'):
    """
    [마스터 버전 - ISO 11783-10 완벽 대응]
    1. AEF Validator 통과: 소수점 9자리 제한 및 꼬리 0 제거 (str 반올림)
    2. FMS 호환: Grid 버퍼 2m 추가, 지수표기법(E-6) 강제 고정, 한글 이름 제거
    3. 기계 호환 (RAUCH 등): TZN J="254" 할당
    4. 단위 표시 완벽화: DDI 0009 (고체 목표량) 적용 및 VPN(kg/ha) 연결
    5. 사용자 편의: _TASKDATA.zip 자동 패키징
    """
    if rasterio is None:
        return None

    taskdata_dir = os.path.join(output_folder, "TASKDATA")
    os.makedirs(taskdata_dir, exist_ok=True)

    gdf_copy = gdf.copy()
    if gdf_copy.crs is None:
        gdf_copy.set_crs("EPSG:5179", inplace=True)
    src_crs = gdf_copy.crs

    gdf_copy['iso_rate'] = gdf_copy[rate_col].round().astype(np.uint32)

    pixel_size = 1.0
    minx, miny, maxx, maxy = gdf_copy.total_bounds

    # [FMS 대응] 필지 이탈 오류 방지 (안전 버퍼 2m)
    padding = 2.0
    minx -= padding
    miny -= padding
    maxx += padding
    maxy += padding

    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    src_transform = from_origin(minx, maxy, pixel_size, pixel_size)

    shapes = ((geom, value) for geom, value in zip(gdf_copy.geometry, gdf_copy['iso_rate']))
    src_array = rasterize(
        shapes,
        out_shape=(height, width),
        transform=src_transform,
        fill=0,
        default_value=0,
        dtype='uint32',
        all_touched=True
    )

    dst_crs = 'EPSG:4326'
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left=minx, bottom=miny, right=maxx, top=maxy
    )

    dst_array = np.zeros((dst_height, dst_width), dtype='uint32')
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest
    )

    bin_array = np.flipud(dst_array)
    bin_path = os.path.join(taskdata_dir, "GRD00000.bin")
    with open(bin_path, 'wb') as f:
        f.write(bin_array.astype('<i4').tobytes())

    min_lon = dst_transform.c
    max_lat = dst_transform.f
    cell_lon = dst_transform.a
    cell_lat = abs(dst_transform.e)
    min_lat = max_lat - (cell_lat * dst_height)

    # [FMS 대응] 파이썬 임의 변환 방지, 무조건 E-6 지수표기법 고정
    cell_lat_str = f"{cell_lat:.15E}".replace('E-0', 'E-').replace('E+0', 'E+')
    cell_lon_str = f"{cell_lon:.15E}".replace('E-0', 'E-').replace('E+0', 'E+')

    # [AEF 및 FMS 대응] 좌표 소수점 9자리 고정 및 불필요한 꼬리 0 제거
    min_lat_str = str(round(min_lat, 9))
    min_lon_str = str(round(min_lon, 9))

    field_area_sqm = int(boundary_geom.area)

    # [FMS 대응] 이름 한글 충돌 방지 및 안전한 ID 생성
    clean_task_name = re.sub(r'[^A-Za-z0-9]', '', task_name)[:15]
    if not clean_task_name:
        clean_task_name = "FIELD"

    unique_suffix = str(random.randint(1000, 9999))
    safe_task_name = f"{clean_task_name}_{unique_suffix}"

    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    boundary_wgs = gpd.GeoSeries([boundary_geom], crs="EPSG:5179").to_crs("EPSG:4326").iloc[0]
    if boundary_wgs.geom_type == 'MultiPolygon':
        poly_to_use = max(boundary_wgs.geoms, key=lambda a: a.area)
    elif boundary_wgs.geom_type == 'Polygon':
        poly_to_use = boundary_wgs
    else:
        poly_to_use = boundary_wgs.convex_hull

    xml_lines = []
    xml_lines.append(
        '<ISO11783_TaskData VersionMinor="0" VersionMajor="4" DataTransferOrigin="1" ManagementSoftwareManufacturer="FMS" TaskControllerManufacturer="FMS" ManagementSoftwareVersion="2.1.6">')
    xml_lines.append('    <CTR A="CTR1" B="daedong"/>')
    xml_lines.append('    <FRM A="FRM1" B="daedong"/>')
    xml_lines.append('    <PDT A="PDT1" B="Fertilizer"/>')
    xml_lines.append(
        f'    <PFD A="PFD1" B="{unique_suffix}" C="{safe_task_name}" D="{field_area_sqm}" E="CTR1" F="FRM1">')
    xml_lines.append(f'        <PLN A="1" B="{safe_task_name}" C="{field_area_sqm}" E="PLN1">')
    xml_lines.append('            <LSG A="1">')

    # [AEF 대응] PNT 좌표도 소수점 9자리 및 꼬리 0 제거
    for lon, lat in zip(*poly_to_use.exterior.coords.xy):
        xml_lines.append(f'                <PNT A="2" C="{str(round(lat, 9))}" D="{str(round(lon, 9))}"/>')

    xml_lines.append('            </LSG>')
    xml_lines.append('        </PLN>')
    xml_lines.append('    </PFD>')
    xml_lines.append(f'    <TSK A="TSK1" B="{safe_task_name}" C="CTR1" D="FRM1" E="PFD1" G="1">')
    xml_lines.append(f'        <TIM A="{now_str}" B="{now_str}" D="1"/>')
    xml_lines.append('        <DLT A="DFFF" B="31"/>')

    # [살포기 호환] RAUCH 등 변량 작동을 위한 J="254" 연결
    xml_lines.append(
        f'        <GRD G="GRD00000" A="{min_lat_str}" B="{min_lon_str}" C="{cell_lat_str}" D="{cell_lon_str}" E="{dst_width}" F="{dst_height}" I="2" J="254"/>')

    # [단위 표시 완벽화] DDI 0009(고체 목표량) 및 화면 표시(VPN1) 연결
    xml_lines.append('        <TZN A="254" B="Default">')
    xml_lines.append('            <PDV A="0009" B="0" C="PDT1" E="VPN1"/>')
    xml_lines.append('        </TZN>')
    xml_lines.append('        <TZN A="253" B="Out of Field">')
    xml_lines.append('            <PDV A="0009" B="0" C="PDT1" E="VPN1"/>')
    xml_lines.append('        </TZN>')
    xml_lines.append('        <TZN A="0" B="Position Lost">')
    xml_lines.append('            <PDV A="0009" B="0" C="PDT1" E="VPN1"/>')
    xml_lines.append('        </TZN>')
    xml_lines.append('    </TSK>')

    # [단위 표시 완벽화] VPN1에 단위 "kg/ha" 강제 주입
    xml_lines.append('    <VPN A="VPN1" B="0" C="1.0" D="0" E="kg/ha"/>')
    xml_lines.append('</ISO11783_TaskData>')

    xml_path = os.path.join(taskdata_dir, "TASKDATA.XML")
    with open(xml_path, 'wb') as f:
        f.write(('\r\n'.join(xml_lines) + '\r\n').encode('utf-8'))

    # [편의성] TASKDATA 자동 ZIP 패키징 (압축 구조 에러 원천 차단)
    taskdata_zip_path = os.path.join(output_folder, f"{safe_task_name}_TASKDATA.zip")
    with zipfile.ZipFile(taskdata_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, arcname="TASKDATA/TASKDATA.XML")
        zf.write(bin_path, arcname="TASKDATA/GRD00000.bin")

    return taskdata_zip_path


def export_csv(gdf, output_folder, file_prefix):
    if gdf.crs is None:
        gdf.set_crs("EPSG:5179", inplace=True)

    centroids_meter = gdf.geometry.centroid
    centroids_wgs = centroids_meter.to_crs(epsg=4326)
    gdf_wgs = gdf.to_crs(epsg=4326)

    gdf_wgs['Center_Lat'] = centroids_wgs.y
    gdf_wgs['Center_Lon'] = centroids_wgs.x

    gdf_wgs['Total_Amount_kg'] = gdf_wgs['F_Total'].round(2)

    vertex_data = []
    max_v = 0

    for idx, row in gdf_wgs.iterrows():
        geom = row['geometry']
        if geom.geom_type == 'Polygon':
            coords = list(geom.exterior.coords)
            if len(coords) > 1 and coords[0] == coords[-1]:
                coords = coords[:-1]
            row_dict = {'grid_id': row['ZONE']}
            for i, (lon, lat) in enumerate(coords):
                row_dict[f'V{i + 1}_Lat'] = lat
                row_dict[f'V{i + 1}_Lon'] = lon
            if len(coords) > max_v: max_v = len(coords)
            vertex_data.append(row_dict)

    v_df = pd.DataFrame(vertex_data)
    out_df = gdf_wgs.drop(columns=['geometry']).merge(v_df, left_on='ZONE', right_on='grid_id', how='left')

    base_cols = ['ZONE', 'PRODUCT', 'DOSE', 'DOSE_UNIT', 'Total_Amount_kg', 'Center_Lat', 'Center_Lon']
    v_cols = []
    for i in range(max_v):
        v_cols.append(f'V{i + 1}_Lat')
        v_cols.append(f'V{i + 1}_Lon')

    final_cols = base_cols + v_cols
    final_cols = [c for c in final_cols if c in out_df.columns]

    csv_path = os.path.join(output_folder, f"{file_prefix}_Result.csv")
    out_df[final_cols].to_csv(csv_path, index=False, encoding='cp949')
    print(f"    - CSV 저장 완료: {os.path.basename(csv_path)}")


def export_dji_tif(gdf, output_folder, file_prefix, rate_col='DOSE'):
    if rasterio is None: return

    src_crs = gdf.crs
    if src_crs is None:
        gdf.set_crs("EPSG:5179", inplace=True)
        src_crs = gdf.crs

    gdf['DJI_Rate'] = gdf[rate_col]

    pixel_size = 1.0
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))

    src_transform = from_origin(minx, maxy, pixel_size, pixel_size)

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

    dst_crs = 'EPSG:4326'
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, left=minx, bottom=miny, right=maxx, top=maxy
    )

    dst_array = np.zeros((dst_height, dst_width), dtype='float32')

    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest
    )

    tif_path = os.path.join(output_folder, f"{file_prefix}_DJI.tif")
    with rasterio.open(
            tif_path, 'w', driver='GTiff',
            height=dst_height, width=dst_width,
            count=1, dtype='float32',
            crs=dst_crs, transform=dst_transform,
            nodata=0
    ) as dst:
        dst.write(dst_array, 1)

    tfw_path = os.path.join(output_folder, f"{file_prefix}_DJI.tfw")
    with open(tfw_path, 'w') as f:
        f.write(f"{dst_transform.a}\n")
        f.write(f"{dst_transform.b}\n")
        f.write(f"{dst_transform.d}\n")
        f.write(f"{dst_transform.e}\n")
        f.write(f"{dst_transform.c + dst_transform.a / 2}\n")
        f.write(f"{dst_transform.f + dst_transform.e / 2}\n")

    print(f"    - DJI 처방맵 저장 완료: {os.path.basename(tif_path)}")


def fix_coordinate_system(gdf, tag):
    if gdf.crs is None:
        gdf.crs = "EPSG:4326"

    bounds = gdf.total_bounds
    mean_x = (bounds[0] + bounds[2]) / 2
    mean_y = (bounds[1] + bounds[3]) / 2

    if (30 < mean_x < 45) and (120 < mean_y < 135):
        gdf['geometry'] = gdf['geometry'].apply(lambda p: Point(p.y, p.x) if p.geom_type == 'Point' else p)

    try:
        return gdf.to_crs(epsg=5179)
    except Exception as e:
        return None


def process_single_field(soil_path, boundary_path, base_name):
    print(f"\n==========================================")
    print(f">>> [필지 처리 시작] {base_name}")
    print(f"==========================================")
    try:
        soil_input = f"zip://{soil_path}".replace("\\", "/") if not soil_path.startswith(
            "zip://") else soil_path.replace("\\", "/")
        bound_input = f"zip://{boundary_path}".replace("\\", "/") if not boundary_path.startswith(
            "zip://") else boundary_path.replace("\\", "/")

        points_gdf = None
        encodings = ['euc-kr', 'cp949', 'latin1', 'utf-8']
        for enc in encodings:
            try:
                points_gdf = gpd.read_file(soil_input, encoding=enc)
                break
            except:
                continue
        if points_gdf is None: points_gdf = gpd.read_file(soil_input)
        boundary_gdf = gpd.read_file(bound_input)

        points_meter = fix_coordinate_system(points_gdf, "토양점")
        boundary_meter = fix_coordinate_system(boundary_gdf, "경계")
        if points_meter is None or boundary_meter is None: return

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

    except Exception as e:
        print(f"[오류] 데이터 로드 실패: {e}")
        return

    boundary_geom = boundary_meter.union_all()
    rotation_angle = get_main_angle(boundary_geom)
    centroid = boundary_geom.centroid
    rotated_boundary = affinity.rotate(boundary_geom, -rotation_angle, origin=centroid)
    xmin, ymin, xmax, ymax = rotated_boundary.bounds

    field_result_dir = os.path.join(RESULT_ROOT, base_name)
    os.makedirs(field_result_dir, exist_ok=True)

    for grid_size in GRID_SIZES:
        file_prefix = f"{base_name}_{grid_size}mx{grid_size}m"
        print(f"\n  ▶ [작업] 그리드 해상도: {grid_size}m x {grid_size}m ({file_prefix})")

        grid_result_dir = os.path.join(field_result_dir, file_prefix)
        os.makedirs(grid_result_dir, exist_ok=True)

        boundary_wgs = boundary_meter.to_crs(epsg=4326)
        boundary_shp_path = os.path.join(grid_result_dir, f"{base_name}_Boundary.shp")
        boundary_wgs.to_file(boundary_shp_path, encoding='utf-8')

        cols = np.arange(xmin, xmax, grid_size)
        rows = np.arange(ymin, ymax, grid_size)
        polygons = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) for x in
                    cols for y in rows]

        temp_grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:5179")
        temp_grid['geometry'] = temp_grid['geometry'].apply(
            lambda g: affinity.rotate(g, rotation_angle, origin=centroid))
        clipped_grid = gpd.clip(temp_grid, boundary_meter).reset_index(drop=True)

        clipped_grid['grid_id'] = (clipped_grid.index + 1).astype(int)
        joined = gpd.sjoin(clipped_grid, points_meter, how="inner", predicate="intersects")
        grid_stats = joined.groupby('grid_id')[analysis_cols].mean().reset_index()
        grid_stats['grid_id'] = grid_stats['grid_id'].astype(int)

        final_grid = clipped_grid.merge(grid_stats, on='grid_id', how='left')
        if final_grid.crs is None:
            final_grid.set_crs(clipped_grid.crs, inplace=True)
        final_grid[analysis_cols] = final_grid[analysis_cols].fillna(0)

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

        total_fertilizer_kg = final_grid['F_Total'].sum()
        print(f"  ----------------------------------------")
        print(f"  ⭐ [{file_prefix}] 필지 총 필요 비료량: {total_fertilizer_kg:,.2f} kg ⭐")
        print(f"  ----------------------------------------")

        raw_dose = final_grid['F_Need_10a'] * 10
        num_classes = 5
        unique_vals = raw_dose.nunique()

        if unique_vals > num_classes:
            _, bins = pd.cut(raw_dose, bins=num_classes, retbins=True, duplicates='drop')
            labels = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
            product_labels = list(range(1, len(labels) + 1))

            final_grid['DOSE'] = pd.cut(raw_dose, bins=bins, labels=labels, include_lowest=True).astype(float).round(2)
            final_grid['PRODUCT'] = pd.cut(raw_dose, bins=bins, labels=product_labels, include_lowest=True).astype(int)
        else:
            final_grid['DOSE'] = raw_dose.round(2)
            final_grid['PRODUCT'] = final_grid['DOSE'].rank(method='dense').astype(int)

        final_grid['ZONE'] = final_grid['grid_id']
        final_grid['DOSE_UNIT'] = 'kg/ha'

        shp_cols = ['DOSE', 'ZONE', 'DOSE_UNIT', 'PRODUCT', 'geometry']
        shp_export_gdf = final_grid[shp_cols].to_crs(epsg=4326)

        output_shp = os.path.join(grid_result_dir, f"{file_prefix}_Result.shp")
        shp_export_gdf.to_file(output_shp, encoding='utf-8')
        print(f"    - SHP 저장 완료: {os.path.basename(output_shp)}")

        zip_path = os.path.join(grid_result_dir, f"{file_prefix}_Result_SHP.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                file_to_zip = output_shp.replace('.shp', ext)
                if os.path.exists(file_to_zip):
                    zf.write(file_to_zip, os.path.basename(file_to_zip))
        print(f"    - 처방맵 ZIP 파일(SHP 포함) 생성 완료: {os.path.basename(zip_path)}")

        export_csv(final_grid, grid_result_dir, file_prefix)
        export_dji_tif(final_grid, grid_result_dir, file_prefix, rate_col='DOSE')

        try:
            zip_out = export_isoxml(final_grid, boundary_geom, grid_result_dir, task_name=file_prefix, rate_col='DOSE')
            if zip_out:
                print(f"    - ISOXML(TASKDATA) ZIP 자동 생성 완료: {os.path.basename(zip_out)}")
        except Exception as e:
            print(f"    - ISOXML 생성 오류: {e}")


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
    if not soil_files: print(f"[알림] 처리할 파일이 없습니다."); return

    print(f"==========================================")
    print(f" 설정: {CROP_TYPE}, 토성: {SOIL_TEXTURE}")
    print(f" 최소시비량: {MIN_N_REQUIREMENT}kg/10a")
    print(f" 다중 해상도 맵 생성: {GRID_SIZES}m")
    print(f"==========================================")

    for soil_file in soil_files:
        matched_boundary, base_name = find_matching_boundary(soil_file, boundary_files)
        if matched_boundary:
            process_single_field(soil_file, matched_boundary, base_name)
        else:
            print(f"\n[건너뜀] 매칭 실패: {os.path.basename(soil_file)}")
    print(f"\n[완료] 결과 폴더: {os.path.abspath(RESULT_ROOT)}")


if __name__ == "__main__":
    main()