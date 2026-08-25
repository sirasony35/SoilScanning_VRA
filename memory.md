# 프로젝트 개요

토양 스캐닝(Soil Scanning) 결과를 입력받아 **가변 시비(VRA, Variable Rate Application) 처방맵**을 자동 생성하는 Python 프로젝트입니다. 필지 경계(Boundary)와 토양 분석 포인트(Shapefile, 주로 유기물 OM 값)를 읽어들여, 필지를 정해진 크기의 격자(grid)로 나누고 각 격자별 토양 상태에 맞는 비료 살포량을 계산합니다. 계산 결과는 농기계(트랙터 FMS, DJI 드론 등)에서 바로 사용할 수 있도록 **ISOXML(ISO 11783) BIN/XML, Shapefile, CSV, GeoTIFF** 등 다양한 포맷으로 출력합니다. 작물(논콩/벼 등)과 토성에 따른 농촌진흥청식 시비 공식을 적용하며, 필지별 목표 비료량 강제 보정 및 토양살균제 혼합 스케일링 기능을 지원합니다.

## 폴더/파일 구조

- `main_grid_vra.py` — 메인 실행 스크립트. 입력 데이터 로드 → 격자 생성 → 시비량 계산 → 처방맵(여러 포맷) 출력까지 전 과정을 수행.
- `fertilizer_calculator.py` — `FertilizerCalculator` 클래스. 작물/토성별 질소 시비 공식을 적용해 격자별 비료 요구량을 산출.
- `demo_vra_manual.py` — **(2026-08-25 신규)** 시연용 처방맵 생성기. 토양 분석 데이터 없이 **경계 폴리곤만으로** 살포폭 격자를 만들고 임의 등급비율·총량 고정으로 처방량을 역산해 출력. 출력 함수는 `main_grid_vra.py` 것을 import 재사용.
- `bin_reader.py` — 디버깅/검증용 유틸. 생성된 ISOXML `GRD00000.bin` 처방량 데이터를 읽어 출력. (독립 실행, 경로 하드코딩됨)
- `FMS/` — 대동(Daedong) FMS(FieldFusion) 농기계 관련 참조용 샘플/출력물.
  - `FMS/TASKDATA/TASKDATA.XML`, `GRD00000.bin` — FMS가 인식하는 ISOXML 처방맵 표준 샘플 (포맷 레퍼런스).
  - `FMS/Prescription/*.shp 등` — 처방 Shapefile 샘플.
  - `FMS/*.zip` — FMS용 ISOXML / Shapefile 처방맵 배포본 예시.
- `new_data/` — 입력 데이터 폴더 예시. `TESTn_Shapefile.zip`(토양 분석 포인트)와 `TESTn_Boundary.zip`(필지 경계) 쌍으로 구성. (`*_data.*`, `*_Boundary.*`는 압축 전 원본)
- `new_data_0416/`, `test_data/` — 다른 시점/지역의 입력 데이터 세트.
- `result/` — 스크립트 실행 결과 출력 폴더(예: `result/TEST1/TEST1_20mx20m/`). 필지별·격자해상도별로 SHP, CSV, DJI TIF, TASKDATA(ISOXML) 등이 생성됨.
- `result_demo/` — `demo_vra_manual.py` 실행 결과 폴더 (예: `result_demo/간척지_비료시연/간척지_비료시연_32mx32m/`).
- `__pycache__/`, `.idea/`, `.git/` — 캐시 / IDE 설정 / git 메타데이터.

## 주요 스크립트 상세

### main_grid_vra.py
- **목적**: 전체 VRA 처방맵 생성 파이프라인의 진입점. 상단 환경 설정값을 바꿔 작물·해상도·출력 포맷을 제어.
- **입력**:
  - `DATA_FOLDER`(코드 상단 설정, 현재 `"real_data"`) 안의 `*_Shapefile.zip`(토양 분석 포인트, 컬럼: `id`, `Longitude`, `Latitude`, `OM(g/kg)`)과 `*_Boundary.zip`(필지 경계 폴리곤) 쌍.
  - 좌표계는 자동 감지/보정(EPSG:4326 ↔ EPSG:5179). 한글 인코딩(euc-kr/cp949 등) 자동 시도.
- **출력**: `RESULT_ROOT`(현재 `"result_gjsm_0625"`) 아래 `<필지명>/<필지명>_<격자>mx<격자>m/` 폴더에:
  - `_Result.shp` + `_Result_SHP.zip` (WGS84 처방 Shapefile, 컬럼 `DOSE/ZONE/DOSE_UNIT/PRODUCT`)
  - `_Result.csv` (cp949 인코딩, ZONE별 처방량·중심좌표·꼭짓점 좌표)
  - `_DJI.tif` + `_DJI.tfw` (DJI 드론용 GeoTIFF 처방맵)
  - `TASKDATA/TASKDATA.XML` + `GRD00000.bin` + `_TASKDATA.zip` (ISOXML GRD 래스터, 트랙터/FMS용)
  - `_Boundary.shp` (필지 경계)
- **핵심 함수**:
  - `main()` — 입력 폴더 스캔, soil/boundary 매칭 후 필지별 처리 루프.
  - `find_matching_boundary()` — Shapefile과 짝이 되는 Boundary zip 매칭.
  - `process_single_field()` — 단일 필지 처리의 핵심. 데이터 로드/정제 → 경계 주축 각도 산출 → 회전 정렬 격자 생성 → 격자별 토양값 공간조인 평균 → `FertilizerCalculator` 호출 → 목표량/살균제 스케일링 → 각종 포맷 출력.
    - **결측 셀 보간(2026-07-27~)**: 공간조인 직후 `NAN_FILL_METHOD == 'idw'`이면 핵심 컬럼에 `fill_nan_by_idw()`를 적용하고, 그 뒤 남은 결측은 기존대로 `fillna(0)` 안전망 처리. `NAN_FILL_METHOD = 'zero'`로 두면 예전(전부 0) 동작으로 복귀.
    - **비료 살포량 스케일링 로직(현재)**: ① `FertilizerCalculator` 산출 격자별 `F_Total` 합계로 시스템 순수 비료량(`total_fertilizer_kg`) 계산. ② `FIELD_TARGET_KG[base_name]`(>0)이 있으면 그 값을, 없으면 시스템 산출량을 기준량(`base_fert_kg`)으로 채택. ③ `total_mixed_kg = base_fert_kg + FIELD_FUNGICIDE_KG[base_name]`(살균제 추가량). ④ `mix_ratio = total_mixed_kg / total_fertilizer_kg`. ⑤ 격자별 처방량은 `F_Need_10a × 10`(10a→ha 환산)에 `mix_ratio`를 곱한 `mixed_dose`로 결정하고, CSV/SHP에 찍히는 `F_Total`도 동일하게 `× mix_ratio`로 갱신. ⑥ `mixed_dose`를 5등급으로 `pd.cut` 분류해 `DOSE`(구간 중앙값)·`PRODUCT`(등급번호) 부여(고유값이 5 이하이면 분류 없이 그대로 사용).
  - `fill_nan_by_idw()` — **(2026-07-27 추가)** 토양 포인트가 하나도 잡히지 않은 결측 격자를 최근접 K개 유효 셀의 IDW(거리 역가중 평균)로 채움. `scipy.spatial.cKDTree`로 셀 중심점 최근접 탐색 → `w = 1/(d^power + eps)` 정규화 가중 평균. 작물별 핵심 컬럼(콩=`OM`, 벼=`OM`+`SI`)에만 적용하며, 결측 셀이 0으로 채워져 시비 공식상 **최대 처방량으로 튀는 문제**를 방지하는 것이 목적. 콘솔에 결측 셀 개수·채운 값 범위를 출력.
  - `get_main_angle()` — 필지의 최소회전사각형으로 주축 각도 계산(격자를 이랑 방향에 맞춰 회전).
  - `fix_coordinate_system()` — 좌표계 자동 판별 및 EPSG:5179(미터) 변환.
  - `export_isoxml()` — 정북형 격자 래스터(.bin) ISOXML 생성. **kg/ha → mg/m² 환산을 위해 처방량 ×100 저장**, VPN `C="0.01"`로 표시 시 환원.
  - `export_isoxml_tzn()` — 폴리곤 TZN(Treatment Zone) 형식 ISOXML 생성(회전 격자 형상 보존, .bin 불필요). 기본 비활성(FMS 미지원).
  - `export_csv()`, `export_dji_tif()` — CSV/GeoTIFF 출력.

### fertilizer_calculator.py
- **목적**: 토양 유기물(OM)·규산(SI)을 기반으로 작물/토성별 질소 시비량과 실제 비료량을 계산.
- **입력**: 격자 GeoDataFrame(`gdf`)과 작물(`crop_type`), 목표수량(`target_yield`), 밑거름 비율, 비료 질소함량, 토성, 최소시비량.
- **출력**: 입력 gdf에 컬럼 추가하여 반환 — `N_Total_Need_10a`(순수 질소 요구량/10a), `N_Basal_10a`, `F_Need_10a`(실제 비료량/10a), `Area_sqm`, `F_Total`(격자별 총 비료 kg).
- **핵심 클래스/로직**: `FertilizerCalculator.execute()`
  - 벼(rice): 목표수량(≥500/≥480/그외)별 `N = a - b×OM + c×규산` 공식, 규산 180 상한·질소 상한 적용.
  - 논콩(soybean): 사질/사양질 `N = 8.178 - 0.232×OM`, 그 외 `N = 9.297 - 0.264×OM`.
  - 밀(wheat)/기타: N=0.
  - 밑거름 비율·최소시비 한계 적용 후 질소함량으로 나눠 실제 비료량 환산.
  - OM/SI 컬럼은 한글·영문 후보로 자동 탐색(`_find_column`), 없으면 기본값(OM 25.0, SI 150.0) 보정.

### demo_vra_manual.py (2026-08-25 신규)
- **목적**: 토양 스캐닝 데이터가 **없는** 필지에서 시연/테스트용 변량시비 맵을 만들 때 사용. `main_grid_vra.py`가 "토양 OM → 시비 공식 → 처방량" 순방향이라면, 이 스크립트는 "**총 살포량 목표 → 등급 비율 → 처방량 역산**" 방향.
- **입력**: 경계 폴리곤 하나(`BOUNDARY_PATH`, zip/shp). 토양 포인트 불필요.
- **핵심 설정**: `WORK_WIDTH`(살포폭=격자 한 변, m), `TOTAL_FERT_KG`(필지 총 살포량), `ZONE_WEIGHTS`(등급 상대비율, 리스트 길이=등급 수), `ZONE_PATTERN`, `SLIVER_RATIO`.
- **핵심 함수**:
  - `build_pass_grid()` — 필지 **장축을 주행방향(x)**, 단축을 **살포폭 방향(y=패스)**으로 잡고 격자 생성. 살포폭 방향의 자투리 패스(폭 < `WORK_WIDTH × SLIVER_RATIO`)는 **인접 패스에 흡수**시켜 주행 불가능한 얇은 띠가 생기지 않게 함. (`main_grid_vra.py`의 단순 `np.arange` 격자는 이 처리가 없어 소필지에서 폭 2~3m짜리 자투리 셀이 생김.)
  - `assign_zones()` — 등급 배치. `'shift'`(패스마다 한 칸씩 밀어 **주행 중 등급이 계속 바뀜**, 시연 추천) / `'stripe'`(패스별 단일 등급, 지면에 띠로 남음) / `'checker'`(인접 칸 항상 다름).
  - `solve_rates()` — **총량 고정 역산**. `Σ(rate_z × area_ha_z) = TOTAL_FERT_KG`를 만족하면서 등급 간 비율이 정확히 `ZONE_WEIGHTS`가 되도록 `k = TOTAL_FERT_KG / Σ(w_z × area_ha_z)`, `rate_z = k × w_z`로 계산. 셀 면적이 클립으로 제각각이어도 총량이 정확히 맞음.
  - `export_all()` — `M.export_isoxml()`, `M.export_csv()`, `M.export_dji_tif()`를 그대로 호출하므로 **FMS 포맷·단위 규약(kg/ha ×100 → mg/m²)은 기존 파이프라인과 100% 동일**.
- **주의**: `M.export_isoxml()`의 `task_name`은 `[^A-Za-z0-9]` 제거 후 15자로 잘리므로 한글 필지명을 넘기면 `FIELD`가 됨 → `TASK_NAME_ASCII`로 영문명을 따로 지정.

### bin_reader.py
- **목적**: 생성된 ISOXML `.bin`(리틀엔디안 32bit 정수, ISO 11783-10 `I="2"`) 파일을 numpy로 읽어 격자 개수·처방량을 확인하는 검증 스크립트.
- **입력**: 하드코딩된 `bin_path`(현재 `result/TEST2_new/.../GRD00000.bin` — 실제 존재하지 않을 수 있음, 사용 시 경로 수정 필요).
- **출력**: 콘솔에 격자 개수와 처방량 배열 출력. (XML의 GRD E/W·F/H 값으로 2D 복원 가능, 주석 처리됨)

## 데이터 흐름 / 실행 순서

1. **입력 준비**: `main_grid_vra.py`의 `DATA_FOLDER`에 `<필지명>_Shapefile.zip`(토양 포인트) + `<필지명>_Boundary.zip`(경계) 쌍을 배치.
2. **설정 조정**: 스크립트 상단에서 `CROP_TYPE`, `TARGET_YIELD`, `SOIL_TEXTURE`, `GRID_SIZES`, 출력 포맷(`EXPORT_ISOXML_GRD/TZN`), 필지별 `FIELD_TARGET_KG`(목표 비료량)·`FIELD_FUNGICIDE_KG`(살균제), 결측 셀 보간 옵션(`NAN_FILL_METHOD`, `IDW_K`, `IDW_POWER`) 등을 지정.
3. **실행**: `python main_grid_vra.py`.
4. **내부 흐름** (필지별):
   - 토양 포인트·경계 로드 → 좌표계 보정(→EPSG:5179) → 컬럼 정제.
   - 경계 주축 각도로 격자를 회전 정렬 후 생성 → 경계로 클립.
   - 각 격자에 포함되는 토양 포인트를 공간조인하여 OM 등 평균값 산출.
   - 포인트가 없는 결측 격자는 IDW로 보간(`NAN_FILL_METHOD='idw'` 기본), 잔여 결측은 0으로 채움.
   - `FertilizerCalculator.execute()`로 격자별 비료 요구량 계산.
   - 필지 목표량/살균제 비율(`mix_ratio`)로 전체 처방량 스케일링, `F_Total` 갱신.
   - 처방량을 5개 등급으로 분류(`DOSE`, `PRODUCT`) → SHP/ZIP, CSV, DJI TIF, ISOXML(GRD/TZN) 출력.
5. **검증(선택)**: 생성된 `GRD00000.bin`을 `bin_reader.py`로 열어 처방량 확인.
6. **활용**: 결과 ZIP을 FMS(트랙터)·DJI(드론) 등 농기계 소프트웨어에 업로드하여 가변 살포.

## 의존성

import 문 기준 사용 라이브러리:
- **geopandas** — Shapefile 입출력, 좌표 변환, 공간 조인/클립.
- **shapely** — 폴리곤/포인트 기하 연산, 회전(affinity).
- **numpy** — 배열·래스터 처리, bin 파일 입출력.
- **pandas** — 데이터프레임, 처방량 구간 분류(`pd.cut`).
- **rasterio** — ISOXML BIN 및 DJI GeoTIFF 생성(래스터화·좌표 재투영). **필수**(없으면 처방맵 생성 불가). `pip install rasterio` 필요.
- **scipy** — `scipy.spatial.cKDTree`로 결측 격자 IDW 보간(`fill_nan_by_idw()` 내부에서 지연 import). `NAN_FILL_METHOD='idw'`(현재 기본값)일 때 **필수**.
- 표준 라이브러리: `math, os, glob, xml.etree.ElementTree, re, datetime, zipfile, random`.

> 설치 예: `pip install geopandas shapely numpy pandas rasterio scipy`

## 현재 설정 스냅샷 (main_grid_vra.py 상단, 2026-07-27 커밋 f6814b4 기준)

| 항목 | 값 |
| --- | --- |
| `DATA_FOLDER` / `RESULT_ROOT` | `real_data` / `result_gjsm_0625` |
| `CROP_TYPE` / `SOIL_TEXTURE` | `soybean`(논콩) / `사질` |
| `TARGET_YIELD` / `BASAL_RATIO` / `MIN_N_REQUIREMENT` | 480 / 100 / 2.0 |
| `FERTILIZER_N_CONTENT` / `FERTILIZER_BAG_WEIGHT` | 0.08(8%) / 20kg |
| `GRID_SIZES` | `[32]` (32m×32m 단일 해상도) |
| `EXPORT_ISOXML_GRD` / `EXPORT_ISOXML_TZN` | True / False |
| `NAN_FILL_METHOD` / `IDW_K` / `IDW_POWER` | `idw` / 5 / 2 |
| `FIELD_TARGET_KG` | SM-1-1·1-2·1-3 = 1500, SM-2-2 = 1360, SM-2-3 = 1340 |
| `FIELD_FUNGICIDE_KG` | SM-1-1 ~ SM-2-3 모두 120 |

→ 즉 현재 코드는 **SM-x 필지 5개, 논콩, 32m 격자, 필지별 목표량 강제 + 살균제 120kg 혼합** 실지 시나리오에 맞춰져 있음().

## 비고

- **실행 환경 (중요)**: `base`/기본 python에는 geopandas가 없음. geopandas + rasterio + scipy가 모두 설치된 conda 환경은 **`python312`**(또는 `pt_model`). 실행: `C:/Users/Sangho/anaconda3/envs/python312/python.exe main_grid_vra.py`. Git Bash 콘솔에서는 한글 출력이 깨져 보이지만 파일 내용은 정상.
- **하드코딩 경로/설정 주의**:
  - `main_grid_vra.py` 상단 `DATA_FOLDER = "real_data"`, `RESULT_ROOT = "result_gjsm_0625"`는 현재 저장소에 실제 `real_data` 폴더가 없음. 동봉된 샘플(`new_data`, `test_data`)로 돌리려면 `DATA_FOLDER` 값을 수정해야 함.
  - `bin_reader.py`의 `bin_path`도 존재하지 않는 경로로 하드코딩되어 있어 직접 수정 후 사용.
  - 입력/출력 경로 모두 **상대 경로**라 스크립트가 있는 폴더에서 실행해야 함.
- **필지명 매핑 의존**: `FIELD_TARGET_KG`, `FIELD_FUNGICIDE_KG` 사전의 키는 입력 파일명(`_Shapefile.zip` 제거한 core name)과 정확히 일치해야 적용됨. 없으면 시스템 산출값/0 사용.
- **단위 규약 (중요)**: ISOXML에서 kg/ha 값을 ISO 표준 mg/m²로 저장하기 위해 처방량에 ×100 후 정수 저장하고, `<VPN C="0.01">`로 표시 시 환원함. GRD/TZN 두 출력 모두 동일 규약.
- **ISOXML 포맷 선택**: `EXPORT_ISOXML_GRD=True`(기본, 정북 래스터, FMS/트랙터 정상 동작), `EXPORT_ISOXML_TZN=False`(폴리곤 TZN, 대동 FMS(FieldFusion) 미지원이라 기본 비활성. 타 ISOXML 호환 SW 테스트 시에만 사용).
- **입력 토양 데이터**: 샘플 Shapefile은 `OM(g/kg)` 컬럼만 포함(762 포인트 등). 규산(SI) 등 다른 항목이 없으면 `FertilizerCalculator`가 기본값으로 보정함 → 논콩(OM 기반) 처방에 최적화된 데이터 구성.
- **인코딩**: 입력은 euc-kr/cp949/latin1/utf-8 순으로 자동 시도, 출력 CSV는 cp949(엑셀 한글 호환), SHP는 utf-8.
- **제조사 메타**: 생성 XML에 `ManagementSoftwareManufacturer="FMS"`, `CTR/FRM B="daedong"`로 대동(Daedong) 농기계 환경을 가정함.
- git 저장소로 이미 관리 중(최근 커밋: "로직 수정" 등). 대용량 결과물(`result/`)과 데이터 zip이 함께 들어있으니 공유 시 `.gitignore` 정리를 고려할 것.

## 변경 이력

- 2026-08-25 **간척지 비료시연 맵 제작** — `test_data/간척지_비료시연.zip`(경계 폴리곤만, 0.5673 ha, EPSG:4326, 전북 부안·김제 간척지 일대 126.687E/35.824N) 대상으로 시연용 처방맵 생성. 토양 데이터가 없어 기존 파이프라인을 못 쓰므로 **`demo_vra_manual.py` 신규 작성**. 설정: 살포폭 32m, 총 200kg(10포대), 3등급 비율 1:2:3, 배치 `shift`. 결과 — 주행방향각 151.2°, 장축 86.4m × 살포폭방향 66.9m, **2패스(32.0/34.9m) × 3구간(32/32/22.4m) = 6칸**, 처방량 **176.07 / 352.14 / 528.21 kg/ha**, 총 200.00 kg. 생성된 `GRD00000.bin` 역검증 결과 라스터 실측 합계 200.01 kg(+0.01%), 필지면적 5,674.8 m²(+0.03%)로 정상. ※ 검증 시 bin 픽셀은 **1 m²가 아니라 약 1.033 m²**(WGS84 재투영 결과, 0.917m × 1.126m)이므로 1px=1m²로 가정하면 총량이 ~3% 적게 나오는 착시가 생김.
- 2026-08-25 재학습 — 커밋 `f6814b4` "로직 수정"(2026-07-27, HEAD) 반영. **결측 격자 IDW 보간 기능 신규 추가**(`fill_nan_by_idw()` 함수 + `NAN_FILL_METHOD`/`IDW_K`/`IDW_POWER` 설정, `process_single_field()`에서 공간조인 직후 호출). 이전에는 토양 포인트가 없는 셀이 `fillna(0)`으로 OM=0이 되어 시비 공식상 최대 처방으로 튀는 문제가 있었음. 이에 따라 **scipy 의존성 추가**. 그 외 로직·설정값 변화 없음(작업트리 clean, main 외 파일 미변경).
- 2026-06-30 pull 반영 — main_grid_vra.py 비료 살포량/로직 수정: `process_single_field`의 살포량 스케일링이 "목표량 기준량 + 살균제 추가량 → 시스템 순수 산출량 대비 `mix_ratio`로 비례 스케일링" 방식으로 정리되고, `FIELD_TARGET_KG` 목표 비료량 값이 갱신됨(SM-1-x 1500, SM-2-2 1360, SM-2-3 1340). `RESULT_ROOT`는 `result_gjsm_0625`로 변경.
