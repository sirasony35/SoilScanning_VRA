import pandas as pd
import numpy as np


class FertilizerCalculator:
    def __init__(self, gdf, crop_type='rice', target_yield=500, basal_ratio=100):
        """
        초기화 함수
        :param gdf: 공간 조인이 완료된 GeoDataFrame
        :param crop_type: 작물 종류 ('rice', 'soybean', 'wheat') - 기본값 'rice'
        :param target_yield: 목표 수량 (단위: kg/10a) - 벼의 경우 500, 480, 460 중 선택
        :param basal_ratio: 밑거름 비율 (0~100) - 예: 100(전량기비) 또는 70(분시)
        """
        self.gdf = gdf.copy()  # 원본 데이터 보호를 위해 복사본 사용
        self.crop_type = crop_type.lower()
        self.target_yield = int(target_yield)
        self.basal_ratio = float(basal_ratio) / 100.0  # 백분율을 소수로 변환 (70 -> 0.7)

        # 분석에 필요한 컬럼 자동 식별
        self.om_col = self._find_column(['OM', 'Om', 'om', 'OrganicMatter'])
        self.si_col = self._find_column(['Si', 'si', 'SiO2', 'sio2', 'Silicate'])

        # 작물별 필수 컬럼 확인
        self._check_required_columns()

    def _find_column(self, candidates):
        """데이터프레임에서 후보군(candidates) 중 존재하는 컬럼명을 찾습니다."""
        for col in candidates:
            if col in self.gdf.columns:
                return col
        return None

    def _check_required_columns(self):
        """작물별로 계산에 꼭 필요한 컬럼이 있는지 확인하고 경고 메시지를 출력합니다."""
        missing = []
        if not self.om_col: missing.append('OM(유기물)')

        # 벼(Rice)는 규산(Si)이 필수
        if self.crop_type == 'rice' and not self.si_col:
            missing.append('Si(유효규산)')

        if missing:
            print(f"  [Calculator] [주의] 필수 데이터가 없습니다: {', '.join(missing)}")
            print("  -> 해당 성분은 0으로 가정하고 계산합니다. 정확도가 떨어질 수 있습니다.")

    def _calculate_rice_formula(self, row):
        """
        [벼(Rice) 전용] 질소 시비량 산출 공식
        """
        # 1. 데이터 가져오기 (없으면 0)
        om = row[self.om_col] if self.om_col else 0
        si_raw = row[self.si_col] if self.si_col else 0

        # 2. 유효규산(Si) 보정: 최대 180으로 제한
        si = min(si_raw, 180)

        n_recommend = 0

        # 3. 목표 수량별 공식 적용
        if self.target_yield >= 500:
            # 목표 500kg: 11.17 - 0.133*OM + 0.025*SiO2
            n_recommend = 11.17 - (0.133 * om) + (0.025 * si)
            # 최대 사용량 제한: 15kg
            n_recommend = min(n_recommend, 15)

        elif self.target_yield >= 480:
            # 목표 480kg: 9.14 - 0.109*OM + 0.020*SiO2
            n_recommend = 9.14 - (0.109 * om) + (0.020 * si)
            # 최대 사용량 제한: 13kg
            n_recommend = min(n_recommend, 13)

        else:  # 460kg 이하 (기본값)
            # 목표 460kg: 7.10 - 0.085*OM + 0.016*SiO2
            n_recommend = 7.10 - (0.085 * om) + (0.016 * si)
            # 별도의 최대 제한 언급이 없으면 계산값 그대로 사용 (필요시 추가 가능)

        # 4. 음수 방지 (최소 0)
        return max(n_recommend, 0)

    def _calculate_soybean_formula(self, row):
        """[논콩 전용] 공식 (추후 구현)"""
        # 현재는 임시로 0 리턴, 추후 공식이 확정되면 구현
        return 0

    def _calculate_wheat_formula(self, row):
        """[밀 전용] 공식 (추후 구현)"""
        # 현재는 임시로 0 리턴
        return 0

    def interpolate_missing_data(self):
        """0값(데이터 공백)을 이웃 그리드 평균으로 보정"""
        zero_indices = self.gdf[self.gdf['N_Need_10a'] == 0].index

        if len(zero_indices) > 0:
            # print(f"  [Calculator] 0값 보정 실행 (대상: {len(zero_indices)}개 셀)")
            for idx in zero_indices:
                current_geom = self.gdf.at[idx, 'geometry']
                neighbors = self.gdf[self.gdf.geometry.touches(current_geom)]
                valid_vals = neighbors[neighbors['N_Need_10a'] > 0]['N_Need_10a']

                if not valid_vals.empty:
                    self.gdf.at[idx, 'N_Need_10a'] = valid_vals.mean()

    def execute(self):
        """
        전체 계산 실행
        """
        print(
            f"  [Calculator] 작물: {self.crop_type.upper()}, 목표수량: {self.target_yield}kg, 밑거름비율: {self.basal_ratio * 100}%")

        # 1. 작물별 총 필요량(Total N) 산출
        if self.crop_type == 'rice':
            self.gdf['N_Total_Need_10a'] = self.gdf.apply(self._calculate_rice_formula, axis=1)
        elif self.crop_type == 'soybean':
            self.gdf['N_Total_Need_10a'] = self.gdf.apply(self._calculate_soybean_formula, axis=1)
        elif self.crop_type == 'wheat':
            self.gdf['N_Total_Need_10a'] = self.gdf.apply(self._calculate_wheat_formula, axis=1)
        else:
            print(f"  [오류] 지원하지 않는 작물입니다: {self.crop_type}")
            self.gdf['N_Total_Need_10a'] = 0

        # 2. 밑거름/분시 비율 적용 (실제 살포할 양 = Rate)
        # N_Need_10a는 기계에 들어갈 최종 '시비율(Rate)'입니다.
        self.gdf['N_Need_10a'] = self.gdf['N_Total_Need_10a'] * self.basal_ratio

        # 3. 0값 보정 (Interpolation)
        self.interpolate_missing_data()

        # 4. 면적 기반 총량(Mass) 계산 (kg)
        self.gdf['Grid_Area'] = self.gdf.geometry.area
        self.gdf['N_Need_area'] = self.gdf['N_Need_10a'] * (self.gdf['Grid_Area'] / 1000)

        # 결과 요약 출력
        avg_rate = self.gdf['N_Need_10a'].mean()
        print(f"  [Calculator] 산출 완료. 평균 시비율: {avg_rate:.2f} kg/10a")

        return self.gdf