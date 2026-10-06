# ISOXML (TASKDATA.XML) 속성 정리

> 생성된 ISOXML 처방맵 파일의 모든 요소·속성을 **ISO 11783-10** 표준 기준으로 정리한 문서입니다.
> 참고 코드: [`main_grid_vra.py`](main_grid_vra.py) → `export_isoxml()` 함수

---

## 0. 전체 구조 개요

```xml
<ISO11783_TaskData>            ← 루트
    <CTR/>                     ← 고객 정보
    <FRM/>                     ← 농장 정보
    <PDT/>                     ← 제품(비료) 정보
    <PFD>                      ← 필지(Partfield)
        <PLN>...</PLN>         ← 필지 경계 폴리곤
    </PFD>
    <TSK>                      ← 작업(Task)
        <TIM/>                 ← 시간
        <DLT/>                 ← 데이터 로깅 트리거
        <GRD/>                 ← 격자 처방 메타데이터
        <TZN>                  ← 처방 zone (Default/OutOfField/Lost)
            <PDV/>             ← 처방값
        </TZN>
    </TSK>
    <VPN/>                     ← 값 표시 형식
</ISO11783_TaskData>
```

---

## 1. 루트: `<ISO11783_TaskData>`

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `VersionMajor` | 표준 메이저 버전 | `4` | ISO 11783-10:4판 |
| `VersionMinor` | 마이너 버전 | `0` | |
| `DataTransferOrigin` | 데이터 발신처 | `1` | 1=FMS, 2=TaskController |
| `ManagementSoftwareManufacturer` | FMS 제조사 | `FMS` | |
| `TaskControllerManufacturer` | 작업기 컨트롤러 제조사 | `FMS` | |
| `ManagementSoftwareVersion` | FMS SW 버전 | `2.1.6` | |

---

## 2. `<CTR>` Customer (고객)

| 속성 | 의미 | 값 |
|---|---|---|
| `A` | CustomerID (고유 ID) | `CTR1` |
| `B` | CustomerDesignator (고객명) | `daedong` |

## 3. `<FRM>` Farm (농장)

| 속성 | 의미 | 값 |
|---|---|---|
| `A` | FarmID | `FRM1` |
| `B` | FarmDesignator (농장명) | `daedong` |

## 4. `<PDT>` Product (제품)

| 속성 | 의미 | 값 |
|---|---|---|
| `A` | ProductID | `PDT1` |
| `B` | ProductDesignator (제품명) | `Fertilizer` |

---

## 5. `<PFD>` Partfield (필지)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | PartfieldID | `PFD1` | 고유 ID |
| `B` | PartfieldCode | `5332` (랜덤 4자리) | 필지 코드 |
| `C` | PartfieldDesignator | `SM2332mx32m` | 필지 표시명 (task_name 정제값) |
| `D` | PartfieldArea | `16215` 등 | 필지 면적 (m²) |
| `E` | CustomerIdRef | `CTR1` | CTR 참조 |
| `F` | FarmIdRef | `FRM1` | FRM 참조 |

---

## 6. `<PLN>` Polygon (필지 경계 폴리곤)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | PolygonType | `1` | **1=PartfieldBoundary**, 2=TreatmentZone, 9=MainfieldBoundary 등 |
| `B` | PolygonDesignator | task 이름 | 표시명 |
| `C` | PolygonArea | 필지 면적 (m²) | |
| `E` | PolygonId | `PLN1` | 폴리곤 고유 ID |

## 7. `<LSG>` Line Segment Group (선 그룹)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | LineStringType | `1` | **1=폴리곤 외곽**, 2=폴리곤 내부 구멍, 3=일반 라인, 5=Track |

## 8. `<PNT>` Point (좌표점)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | PointType | `2` | 1=Flag, **2=Other(범용 좌표)**, 6=GuidanceA, 10=PartfieldRef 등 |
| `C` | PointNorth (위도) | `35.825...` | WGS84 십진도 |
| `D` | PointEast (경도) | `126.683...` | WGS84 십진도 |

---

## 9. `<TSK>` Task (작업)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | TaskID | `TSK1` | 작업 고유 ID |
| `B` | TaskDesignator | `SM2332mx32m` | 작업명 |
| `C` | CustomerIdRef | `CTR1` | |
| `D` | FarmIdRef | `FRM1` | |
| `E` | PartfieldIdRef | `PFD1` | 대상 필지 |
| `G` | TaskStatus | `1` | **1=Planned**, 2=Running, 3=Paused, 4=Completed, 5=Template, 6=Canceled |

## 10. `<TIM>` Time (작업 시간)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | StartTime | `2026-06-25T...Z` | UTC ISO 8601 |
| `B` | StopTime | (동일) | |
| `D` | TimeType | `1` | **1=Planned**, 4=Effective, 5=Ineffective, 6=Repair |

## 11. `<DLT>` DataLogTrigger (로깅 트리거)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | DataLogDDI | `DFFF` | 로깅할 데이터 DDI (DFFF = 사용자 정의/모든 DDI) |
| `B` | DataLogMethod | `31` | 비트마스크: 1=Time, 2=Distance, 4=Threshold, 8=OnChange, 16=Total. **31 = 모든 트리거 켜짐** |

---

## 12. `<GRD>` Grid (격자 처방 메타데이터) — **핵심**

| 속성 | 의미 | 예시 값 | 설명 |
|---|---|---|---|
| `G` | GridFilename | `GRD00000` | .bin 파일 이름 (확장자 없이) |
| `A` | GridMinimumNorth | `35.825...` | 격자 남쪽 끝 위도 |
| `B` | GridMinimumEast | `126.683...` | 격자 서쪽 끝 경도 |
| `C` | GridCellNorthSize | `9.04E-6` | **셀 하나의 위도 간격(도)** ≈ 1m |
| `D` | GridCellEastSize | `1.10E-5` | **셀 하나의 경도 간격(도)** ≈ 1m |
| `E` | GridMaximumColumn | `42` | 가로 픽셀 개수 |
| `F` | GridMaximumRow | `58` | 세로 픽셀 개수 |
| `I` | GridType | `2` | **2 = TreatmentZoneIdReferenced (32bit 정수, .bin에 처방값 직접 저장)**, 1 = TreatmentZoneIdOnly |
| `J` | TreatmentZoneCode | `254` | 처방값이 없거나 매칭 안 될 때 사용할 기본 TZN 코드 |

> 📌 `pixel_size = 1.0` ([main_grid_vra.py:98](main_grid_vra.py)) 설정이 여기 C/D 값을 결정합니다.

---

## 13. `<TZN>` TreatmentZone (처방 zone)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | TreatmentZoneCode | `254` / `253` / `0` | 1~254 사용 가능. 특별 코드: **254=Default**, 253=OutOfField, 0=PositionLost |
| `B` | TreatmentZoneDesignator | `Default` 등 | zone 이름 |
| `C` | (선택) Colour | 미설정 | 0~254 색상 인덱스 |

GRD 방식에서는 위 3개 zone만 정의하고, 실제 처방값은 .bin에 32비트 정수로 저장됩니다.

## 14. `<PDV>` ProcessDataValue (처방값) — **핵심**

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | ProcessDataDDI | `0009` | DDI (Data Dictionary ID, 4자리 16진수). **0x0006이 표준** (Setpoint Mass per Area Rate) |
| `B` | ProcessDataValue | `0` (Default zone) | 처방값 raw 정수 (mg/m² 단위) |
| `C` | ProductIdRef | `PDT1` | PDT 참조 |
| `E` | ValuePresentationIdRef | `VPN1` | VPN 참조 (표시 형식) |

---

## 15. `<VPN>` ValuePresentation (값 표시 형식)

| 속성 | 의미 | 현재 값 | 설명 |
|---|---|---|---|
| `A` | ValuePresentationID | `VPN1` | ID |
| `B` | Offset | `0` | 표시값 = (raw × Scale) + Offset |
| `C` | Scale | `0.01` | **raw 100을 표시 0.01단위로** = mg/m² → kg/ha 환원 |
| `D` | NumberOfDecimals | `2` | 소수점 자릿수 |
| `E` | UnitDesignator | `kg/ha` | 단위 텍스트 |

> 📌 raw 88,600 (mg/m²) × 0.01 = **886.00 kg/ha**로 표시되는 메커니즘.

---

## 16. 처방값 흐름 요약 (단위 환산 전체 그림)

```
파이썬 final_grid['DOSE'] = 886.0 (kg/ha)
            ↓ × 100 (main_grid_vra.py:97)
.bin 파일에 저장 = 88600 (정수, mg/m²)
            ↓ 트랙터 컨트롤러 읽음 (DDI 0x0009 → mg/m² 해석)
머신 내부 처리 = 88600 mg/m² = 886 kg/ha
            ↓ FMS 표시 (VPN scale 0.01 적용)
화면 표시 = 886.00 kg/ha
```

---

## 17. 참고 — 사용된 DDI / 사용하지 않은 속성

### 사용 중인 DDI

| DDI | 위치 | 용도 |
|---|---|---|
| `0009` | PDV `A` 속성 | 현재 처방 rate (FMS가 너그럽게 수용 중) |
| `DFFF` | DLT `A` 속성 | 로깅 와일드카드 (모든 DDI) |

### ISO 표준 권장 DDI (참고)

| DDI | 의미 | 용도 |
|---|---|---|
| `0006` | Setpoint Mass per Area Application Rate (mg/m²) | **비료 처방 표준** ⭐ |
| `0001` | Setpoint Volume per Area Application Rate (mL/m²) | 액제 처방 |
| `0005` | Setpoint Count per Area Application Rate | 종자 처방 |

### 미사용 속성 (확장 여지)

| 요소 | 미사용 속성 | 의미 |
|---|---|---|
| TZN | `C` | Colour (FMS에서 zone별 색 지정 가능) |
| PNT | `E` / `H` / `I` | Up(고도), 수평·수직 정밀도 |
| TSK | `F` / `H` | ResponsibleWorkerRef, DefaultTreatmentZoneCode |
| PFD | `G` / `H` / `I` | CropTypeRef, CropVarietyRef, FieldRef |

---

## 부록 A — 두 종류 셀의 구분

ISOXML 출력에서 "셀"이라는 단어는 두 가지 다른 의미로 쓰이므로 혼동 주의.

| 구분 | 코드 위치 | 크기 | 역할 |
|---|---|---|---|
| **처방 셀** (Prescription Cell) | [`main_grid_vra.py:42`](main_grid_vra.py) `GRID_SIZES = [32]` | **32m × 32m** | 같은 처방량(DOSE)이 적용되는 농업적 단위 |
| **ISOXML 픽셀** (GRD Raster Cell) | [`main_grid_vra.py:98`](main_grid_vra.py) `pixel_size = 1.0` | **1m × 1m** | ISO 11783 GRD 표준 래스터의 저장 단위 |

하나의 32m × 32m 처방 셀 안에 **32 × 32 = 1,024개**의 ISOXML 픽셀이 들어가며, 같은 처방량 값을 공유합니다.

## 부록 B — pixel_size 변경 시 영향

| 값 | 외곽 표현 | 파일 크기 (1.6ha 기준) | 처방 셀당 픽셀 수 |
|---|---|---|---|
| `1.0` (현재) | 거친 계단형 | ~150 KB | 1,024 |
| `0.5` | 부드러움 | ~600 KB | 4,096 |
| `0.25` | 매우 부드러움 | ~2.4 MB | 16,384 |
| `32.0` | 처방 셀 = 픽셀 (1:1) | ~150 B | 1 (단, 회전 셀이 정북 정사각으로 변형되어 매핑 어긋남) |

---

*문서 생성일: 2026-06-25 · 작성자: Claude*
