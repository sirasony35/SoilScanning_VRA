import pandas as pd
import numpy as np

class FertilizerCalculator:
    def __init__(self, gdf, crop_type='rice', target_yield=500, basal_ratio=100,
                 fertilizer_n_content=0.20, soil_texture='식양질', min_n_limit=2.0):
        """
        초기화 함수
        """
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

        # [수정] 컬럼 찾기 기능 강화
        self.om_col = self._find_column(['OM', 'Om', 'om', 'OrganicMatter'])
        self.si_col = self._find_column(['Si', 'si', 'SiO2', 'sio2', 'Silicate'])

        # [디버깅] 찾은 컬럼 확인 출력
        if self.om_col:
            print(f"  [Calculator] 유기물(OM) 데이터 컬럼 인식됨: '{self.om_col}'")
        else:
            print("  [Calculator] [경고] 유기물(OM) 컬럼을 찾을 수 없습니다! 계산 시 0으로 처리됩니다.")

        self._check_required_columns()

    def _find_column(self, candidates):
        """
        데이터프레임에서 후보군(candidates) 중 존재하는 컬럼명을 찾습니다.
        대소문자 및 공백을 무시하고 비교합니다.
        """
        # 데이터프레임의 실제 컬럼들을 정리 (대문자+공백제거)하여 매핑
        clean_cols = {col.strip().upper(): col for col in self.gdf.columns}

        for candidate in candidates:
            cand_clean = candidate.strip().upper()
            if cand_clean in clean_cols:
                return clean_cols[cand_clean]  # 실제 컬럼명 반환
        return None

    def _check_required_columns(self):
        missing = []
        if not self.om_col: missing.append('OM(유기물)')
        if self.crop_type == 'rice' and not self.si_col:
            missing.append('Si(유효규산)')
        if missing:
            print(f"  [Calculator] [주의] 필수 데이터가 없습니다: {', '.join(missing)}")

    def _calculate_rice_formula(self, row):
        om = row[self.om_col] if self.om_col else 0
        si_raw = row[self.si_col] if self.si_col else 0
        si = min(si_raw, 180)

        n_recommend = 0
        if self.target_yield >= 500:
            n_recommend = 11.17 - (0.133 * om) + (0.025 * si)
            n_recommend = min(n_recommend, 15)
        elif self.target_yield >= 480:
            n_recommend = 9.14 - (0.109 * om) + (0.020 * si)
            n_recommend = min(n_recommend, 13)
        else:
            n_recommend = 7.10 - (0.085 * om) + (0.016 * si)

        return max(n_recommend, 0)

    def _calculate_soybean_formula(self, row):
        om = row[self.om_col] if self.om_col else 0
        sandy_types = ['사질', '사양질', 'sandy', 'sandyloam']

        if self.soil_texture in sandy_types:
            n_recommend = 8.178 - (0.232 * om)
        else:
            n_recommend = 9.297 - (0.264 * om)

        return max(n_recommend, 0)

    def _calculate_wheat_formula(self, row):
        return 0

    def interpolate_missing_data(self):
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
        zero_mask = self.gdf['N_Need_10a'] < self.min_n_limit
        count = zero_mask.sum()
        if count > 0:
            print(f"  [Calculator] 최소 기준({self.min_n_limit}kg) 미달 {count}개 셀을 최소값으로 보정합니다.")
            self.gdf.loc[zero_mask, 'N_Need_10a'] = self.min_n_limit

    def execute(self):
        print(f"  [Calculator] 작물: {self.crop_type.upper()}, 토성: {self.soil_texture}")
        print(f"  [Calculator] N함량: {self.fertilizer_n_content * 100}%, 최소시비량설정: {self.min_n_limit}kg/10a")

        # 1. 순수 질소 요구량 산출
        if self.crop_type == 'rice':
            self.gdf['N_Total_Need_10a'] = self.gdf.apply(self._calculate_rice_formula, axis=1)
        elif self.crop_type == 'soybean':
            self.gdf['N_Total_Need_10a'] = self.gdf.apply(self._calculate_soybean_formula, axis=1)
        elif self.crop_type == 'wheat':
            self.gdf['N_Total_Need_10a'] = self.gdf.apply(self._calculate_wheat_formula, axis=1)
        else:
            self.gdf['N_Total_Need_10a'] = 0

        # 2. 밑거름 비율 적용
        self.gdf['N_Need_10a'] = self.gdf['N_Total_Need_10a'] * self.basal_ratio

        # 3. 보정 (반복 보간법)
        self.interpolate_missing_data()

        # 4. 최소 시비량 적용
        self.apply_minimum_floor()

        # 5. 순수 질소 총량
        self.gdf['Grid_Area'] = self.gdf.geometry.area
        self.gdf['N_Total'] = self.gdf['N_Need_10a'] * (self.gdf['Grid_Area'] / 1000)

        # 6. 실제 비료량 환산
        self.gdf['F_Need_10a'] = self.gdf['N_Need_10a'] / self.fertilizer_n_content
        self.gdf['F_Total'] = self.gdf['F_Need_10a'] * (self.gdf['Grid_Area'] / 1000)

        avg_f_rate = self.gdf['F_Need_10a'].mean()
        print(f"  [Calculator] 산출 완료. 평균 비료 시비율: {avg_f_rate:.2f} kg/10a")

        return self.gdf