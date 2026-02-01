import pandas as pd
import numpy as np


class FertilizerCalculator:
    def __init__(self, gdf, crop_type='rice', target_yield=500, basal_ratio=100,
                 fertilizer_n_content=0.20, soil_texture='식양질', min_n_limit=2.0):
        """
        초기화 함수
        :param min_n_limit: 계산값이 0일 때 적용할 최소 질소 시비량 (기본값: 2.0kg/10a)
        """
        self.gdf = gdf.copy()
        self.crop_type = crop_type.lower()
        self.target_yield = int(target_yield)
        self.basal_ratio = float(basal_ratio) / 100.0
        self.fertilizer_n_content = float(fertilizer_n_content)
        self.soil_texture = str(soil_texture).strip().replace(" ", "").lower()

        # [신규] 최소 시비량 설정 (0 방지용)
        self.min_n_limit = float(min_n_limit)

        if self.fertilizer_n_content <= 0:
            print("  [Calculator] [경고] 비료 질소 함량이 0 이하입니다. 1.0(100%)으로 강제 설정합니다.")
            self.fertilizer_n_content = 1.0

        self.om_col = self._find_column(['OM', 'Om', 'om', 'OrganicMatter'])
        self.si_col = self._find_column(['Si', 'si', 'SiO2', 'sio2', 'Silicate'])
        self._check_required_columns()

    def _find_column(self, candidates):
        for col in candidates:
            if col in self.gdf.columns:
                return col
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
        """
        [수정됨] 0값(데이터 공백)을 이웃 그리드 평균으로 보정
        - 반복문(loop)을 사용하여 뭉쳐있는 0값들을 순차적으로 채웁니다.
        """
        # 최대 5번 반복하여 안쪽 구멍까지 값을 채움
        for i in range(5):
            zero_indices = self.gdf[self.gdf['N_Need_10a'] <= 0].index  # 0 이하인 값 대상

            if len(zero_indices) == 0:
                break  # 0인 값이 없으면 중단

            filled_count = 0
            for idx in zero_indices:
                current_geom = self.gdf.at[idx, 'geometry']
                # 맞닿은 이웃 찾기
                neighbors = self.gdf[self.gdf.geometry.touches(current_geom)]
                # 유효한(0보다 큰) 이웃 값만 추출
                valid_vals = neighbors[neighbors['N_Need_10a'] > 0]['N_Need_10a']

                if not valid_vals.empty:
                    # 이웃 평균으로 대체
                    self.gdf.at[idx, 'N_Need_10a'] = valid_vals.mean()
                    filled_count += 1

            # 더 이상 채워진 게 없으면 중단 (무한루프 방지)
            if filled_count == 0:
                break

    def apply_minimum_floor(self):
        """
        [신규] 보정 후에도 여전히 0인 값(전체가 0인 경우 등)에 최소 시비량 적용
        """
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

        # 4. [신규] 최소 시비량 적용 (최후의 수단)
        # 만약 주변도 다 0이라서 보정이 안 되었다면, 최소값(예: 2kg)을 강제 적용
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