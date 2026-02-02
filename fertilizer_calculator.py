import pandas as pd
import numpy as np


class FertilizerCalculator:
    def __init__(self, gdf, crop_type='rice', target_yield=500, basal_ratio=100,
                 fertilizer_n_content=0.20, soil_texture='식양질', min_n_limit=2.0):
        self.gdf = gdf.copy()
        self.crop_type = crop_type.lower()
        self.target_yield = int(target_yield)
        self.basal_ratio = float(basal_ratio) / 100.0
        self.fertilizer_n_content = float(fertilizer_n_content)
        self.soil_texture = str(soil_texture).strip().replace(" ", "").lower()
        self.min_n_limit = float(min_n_limit)

        if self.fertilizer_n_content <= 0:
            print("  [Calculator] [경고] 비료 질소 함량이 0 이하입니다. 1.0(100%)으로 강제 설정합니다.")
            self.fertilizer_n_content = 1.0

        self.om_col = self._find_column(['OM', 'Om', 'om', 'OrganicMatter'])
        self.si_col = self._find_column(['Si', 'si', 'SiO2', 'sio2', 'Silicate'])
        self._check_required_columns()

    def _find_column(self, candidates):
        clean_cols = {col.strip().upper(): col for col in self.gdf.columns}
        for candidate in candidates:
            cand_clean = candidate.strip().upper()
            if cand_clean in clean_cols:
                return clean_cols[cand_clean]
        return None

    def _check_required_columns(self):
        missing = []
        if not self.om_col: missing.append('OM(유기물)')
        if self.crop_type == 'rice' and not self.si_col:
            missing.append('Si(유효규산)')
        if missing:
            print(f"  [Calculator] [주의] 필수 데이터가 없습니다: {', '.join(missing)}")

    def interpolate_soil_data(self):
        """
        [신규] 토양 성분(OM, Si)이 0인 곳을 주변 유효값 평균으로 채웁니다.
        계산 결과(N량)가 아니라 입력 데이터(토양)를 보정하므로 더 정확합니다.
        """
        # 보정할 컬럼 목록 선정
        cols_to_fix = []
        if self.om_col:
            cols_to_fix.append(self.om_col)

        # 벼인 경우 규산(Si)도 보정 대상
        if self.crop_type == 'rice' and self.si_col:
            cols_to_fix.append(self.si_col)

        if not cols_to_fix:
            return

        print(f"  [Calculator] 토양 데이터 보정 시작 (대상: {cols_to_fix})")

        # 각 컬럼별로 보정 수행 (최대 3회 반복하여 구멍 메우기)
        for col in cols_to_fix:
            # 해당 컬럼을 float형으로 확실히 변환 (에러 방지)
            self.gdf[col] = pd.to_numeric(self.gdf[col], errors='coerce').fillna(0)

            for i in range(3):
                # 값이 0이거나 NaN인 인덱스 찾기
                zero_indices = self.gdf[self.gdf[col] <= 0].index
                if len(zero_indices) == 0:
                    break  # 보정할 게 없으면 중단

                filled_count = 0
                for idx in zero_indices:
                    current_geom = self.gdf.at[idx, 'geometry']
                    # 맞닿은 이웃 찾기
                    neighbors = self.gdf[self.gdf.geometry.touches(current_geom)]
                    # 이웃 중 값이 0보다 큰 것만 추출
                    valid_vals = neighbors[neighbors[col] > 0][col]

                    if not valid_vals.empty:
                        self.gdf.at[idx, col] = valid_vals.mean()
                        filled_count += 1

                # print(f"    - {col} 보정 round {i+1}: {filled_count}개 셀 채움")
                if filled_count == 0:
                    break

    def apply_minimum_floor(self):
        """최소 시비량 강제 적용 (계산된 N요구량이 너무 작을 때 기계 한계치 적용)"""
        mask = self.gdf['N_Need_10a'] < self.min_n_limit
        count = mask.sum()
        if count > 0:
            # print(f"  [Calculator] 최소 기준({self.min_n_limit}kg) 미달 {count}개 셀을 최소값으로 설정")
            self.gdf.loc[mask, 'N_Need_10a'] = self.min_n_limit

    def execute(self):
        print(f"  [Calculator] 작물: {self.crop_type.upper()}, 토성: {self.soil_texture}")

        # 1. [신규] 토양 데이터 선행 보정
        # (비료 계산 전에 빈 토양 데이터를 채웁니다)
        self.interpolate_soil_data()

        # [벡터화 연산] 데이터 준비
        om_series = self.gdf[self.om_col].fillna(0).astype(float) if self.om_col else pd.Series(0, index=self.gdf.index)
        si_series = self.gdf[self.si_col].fillna(0).astype(float) if self.si_col else pd.Series(0, index=self.gdf.index)

        # [디버깅] 값 확인
        print(f"  [검증] 보정 후 OM 평균: {om_series.mean():.4f}")

        # 규산 최대값 제한
        si_series = si_series.clip(upper=180)

        # 2. 순수 질소 요구량 산출
        if self.crop_type == 'rice':
            if self.target_yield >= 500:
                n_calc = 11.17 - (0.133 * om_series) + (0.025 * si_series)
                n_calc = n_calc.clip(upper=15)
            elif self.target_yield >= 480:
                n_calc = 9.14 - (0.109 * om_series) + (0.020 * si_series)
                n_calc = n_calc.clip(upper=13)
            else:
                n_calc = 7.10 - (0.085 * om_series) + (0.016 * si_series)

            self.gdf['N_Total_Need_10a'] = n_calc.clip(lower=0)

        elif self.crop_type == 'soybean':
            if self.soil_texture in ['사질', '사양질', 'sandy', 'sandyloam']:
                n_calc = 8.178 - (0.232 * om_series)
            else:
                # 논콩(그외) 공식
                n_calc = 9.297 - (0.264 * om_series)

            # 음수 제거 (매우 중요: 유기물이 높아서 음수가 나오면 0으로 처리됨)
            self.gdf['N_Total_Need_10a'] = n_calc.clip(lower=0)

        elif self.crop_type == 'wheat':
            self.gdf['N_Total_Need_10a'] = 0
        else:
            self.gdf['N_Total_Need_10a'] = 0

        # 3. 밑거름 비율 적용
        self.gdf['N_Need_10a'] = self.gdf['N_Total_Need_10a'] * self.basal_ratio

        # (기존 N값 보정 로직인 interpolate_missing_data는 삭제되었습니다)

        # 4. 최소 시비량 적용
        # (계산 결과가 0인 곳은 여기서 최소량으로 살아납니다)
        self.apply_minimum_floor()

        # 5. 순수 질소 총량 & 실제 비료량 환산
        self.gdf['Grid_Area'] = self.gdf.geometry.area
        self.gdf['N_Total'] = self.gdf['N_Need_10a'] * (self.gdf['Grid_Area'] / 1000)
        self.gdf['F_Need_10a'] = self.gdf['N_Need_10a'] / self.fertilizer_n_content
        self.gdf['F_Total'] = self.gdf['F_Need_10a'] * (self.gdf['Grid_Area'] / 1000)

        avg_f_rate = self.gdf['F_Need_10a'].mean()
        print(f"  [Calculator] 산출 완료. 평균 비료 시비율: {avg_f_rate:.2f} kg/10a")

        return self.gdf