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

    def interpolate_missing_data(self):
        """
        0값(데이터 공백)을 이웃 그리드 평균으로 보정
        [수정] 논콩(Soybean)은 0이 정상적인 값일 수 있으므로 보정을 건너뜁니다.
        """
        if self.crop_type == 'soybean':
            print("  [Calculator] 논콩 작물은 0값 보정(Interpolation)을 수행하지 않습니다. (0=정상)")
            return

        for i in range(5):
            zero_indices = self.gdf[self.gdf['N_Need_10a'] <= 0].index
            if len(zero_indices) == 0: break
            filled_count = 0
            for idx in zero_indices:
                current_geom = self.gdf.at[idx, 'geometry']
                neighbors = self.gdf[self.gdf.geometry.touches(current_geom)]
                valid_vals = neighbors[neighbors['N_Need_10a'] > 0]['N_Need_10a']
                if not valid_vals.empty:
                    self.gdf.at[idx, 'N_Need_10a'] = valid_vals.mean()
                    filled_count += 1
            if filled_count == 0: break

    def apply_minimum_floor(self):
        """최소 시비량 강제 적용"""
        mask = self.gdf['N_Need_10a'] < self.min_n_limit
        count = mask.sum()
        if count > 0:
            print(f"  [Calculator] 최소 기준({self.min_n_limit}kg) 미달 {count}개 셀을 최소값으로 설정")
            self.gdf.loc[mask, 'N_Need_10a'] = self.min_n_limit

    def execute(self):
        print(f"  [Calculator] 작물: {self.crop_type.upper()}, 토성: {self.soil_texture}")

        # [벡터화 연산] 데이터 준비
        om_series = self.gdf[self.om_col].fillna(0).astype(float) if self.om_col else pd.Series(0, index=self.gdf.index)
        si_series = self.gdf[self.si_col].fillna(0).astype(float) if self.si_col else pd.Series(0, index=self.gdf.index)

        # [디버깅] 값 확인
        print(f"  [검증-1] 입력 OM 평균: {om_series.mean():.4f}")

        # 규산 최대값 제한
        si_series = si_series.clip(upper=180)

        # 1. 순수 질소 요구량 산출
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

            # 음수 제거 (매우 중요: 여기서 0이 됨)
            self.gdf['N_Total_Need_10a'] = n_calc.clip(lower=0)
            print(f"  [검증-2] 1차 계산된 N요구량 평균: {self.gdf['N_Total_Need_10a'].mean():.4f}")

        elif self.crop_type == 'wheat':
            self.gdf['N_Total_Need_10a'] = 0
        else:
            self.gdf['N_Total_Need_10a'] = 0

        # 2. 밑거름 비율 적용
        self.gdf['N_Need_10a'] = self.gdf['N_Total_Need_10a'] * self.basal_ratio

        # 3. 보정 (반복 보간법) -> 논콩은 건너뜀!
        self.interpolate_missing_data()

        # 4. 최소 시비량 적용 (여기서 0인 값들이 2.0으로 살아나야 함)
        self.apply_minimum_floor()

        # 5. 순수 질소 총량 & 실제 비료량 환산
        self.gdf['Grid_Area'] = self.gdf.geometry.area
        self.gdf['N_Total'] = self.gdf['N_Need_10a'] * (self.gdf['Grid_Area'] / 1000)
        self.gdf['F_Need_10a'] = self.gdf['N_Need_10a'] / self.fertilizer_n_content
        self.gdf['F_Total'] = self.gdf['F_Need_10a'] * (self.gdf['Grid_Area'] / 1000)

        avg_f_rate = self.gdf['F_Need_10a'].mean()
        print(f"  [Calculator] 산출 완료. 평균 비료 시비율: {avg_f_rate:.2f} kg/10a")

        return self.gdf