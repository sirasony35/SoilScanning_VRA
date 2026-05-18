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

        # 한글, 영문 등 다양한 케이스 후보 등록
        self.om_col = self._find_column(['OM', 'OrganicMatter', '유기물'])
        self.si_col = self._find_column(['SI', 'SiO2', 'Silicate', '규산'])
        self._check_required_columns()

    def _find_column(self, candidates):
        for col in self.gdf.columns:
            # 🚨 [추가] 지형 데이터 컬럼(geometry)은 'OM'이 포함되어 있어도 무조건 건너뜀
            if col.upper() == 'GEOMETRY':
                continue

            for cand in candidates:
                if cand.upper() in col.upper():
                    return col
        return None

    def _check_required_columns(self):
        if not self.om_col:
            self.gdf['OM_임시'] = 25.0
            self.om_col = 'OM_임시'
        if not self.si_col:
            self.gdf['SI_임시'] = 150.0
            self.si_col = 'SI_임시'

    def execute(self):
        om_series = self.gdf[self.om_col].fillna(0)
        si_series = self.gdf[self.si_col].fillna(0)

        print("  ----------------------------------------")
        print(f"  🌱 [시비량 산출 엔진 가동] 작물: {self.crop_type.upper()} / 토성: {self.soil_texture}")

        if self.crop_type == 'rice':
            if self.soil_texture in ['사질', '사양질', 'sandy', 'sandyloam']:
                n_calc = 9.14 - (0.109 * om_series) + (0.020 * si_series)
                n_calc = n_calc.clip(upper=13)
                print("  💡 [적용 공식] 벼 (사질/사양질) : 순수 질소(N) = 9.14 - (0.109 × 유기물) + (0.020 × 유효규산)")
            else:
                n_calc = 7.10 - (0.085 * om_series) + (0.016 * si_series)
                print("  💡 [적용 공식] 벼 (일반/식양질) : 순수 질소(N) = 7.10 - (0.085 × 유기물) + (0.016 × 유효규산)")

            self.gdf['N_Total_Need_10a'] = n_calc.clip(lower=0)

        elif self.crop_type == 'soybean':
            if self.soil_texture in ['사질', '사양질', 'sandy', 'sandyloam']:
                n_calc = 8.178 - (0.232 * om_series)
                print("  💡 [적용 공식] 논콩 (사질/사양질) : 순수 질소(N) = 8.178 - (0.232 × 유기물)")
            else:
                n_calc = 9.297 - (0.264 * om_series)
                print("  💡 [적용 공식] 논콩 (일반/식양질) : 순수 질소(N) = 9.297 - (0.264 × 유기물)")

            self.gdf['N_Total_Need_10a'] = n_calc.clip(lower=0)

        elif self.crop_type == 'wheat':
            print("  💡 [적용 공식] 밀 : N = 기본 시비량 적용 (현재 0으로 세팅됨)")
            self.gdf['N_Total_Need_10a'] = 0
        else:
            print("  💡 [적용 공식] 미지원 작물 : N = 0")
            self.gdf['N_Total_Need_10a'] = 0

        # 3. 밑거름 비율 적용
        self.gdf['N_Basal_10a'] = self.gdf['N_Total_Need_10a'] * self.basal_ratio

        # 최소 시비 한계값 적용
        self.gdf['N_Basal_10a'] = self.gdf['N_Basal_10a'].clip(lower=self.min_n_limit)

        # 4. 순수 질소량 -> 실제 비료량 환산 (질소 함량 나누기)
        self.gdf['F_Need_10a'] = self.gdf['N_Basal_10a'] / self.fertilizer_n_content

        print(f"  ✔️ 적용된 비료 질소 함량: {self.fertilizer_n_content * 100:.1f}%")
        print(f"  ✔️ 적용된 밑거름 비율: {self.basal_ratio * 100:.1f}%")
        print(f"  ✔️ 최소 시비 한계(N): {self.min_n_limit} kg/10a")

        # 5. F_Total 계산 (그리드별 총 필요량)
        self.gdf['Area_sqm'] = self.gdf.geometry.area
        self.gdf['F_Total'] = self.gdf['F_Need_10a'] * (self.gdf['Area_sqm'] / 1000.0)

        return self.gdf