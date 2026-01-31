import pandas as pd
import numpy as np


class FertilizerCalculator:
    def __init__(self, gdf):
        """
        초기화 함수
        :param gdf: 공간 조인이 완료된 GeoDataFrame
        """
        self.gdf = gdf
        self.om_col = self._find_om_column()

    def _find_om_column(self):
        """OM(유기물) 컬럼을 대소문자 구분 없이 찾습니다."""
        if 'OM' in self.gdf.columns:
            return 'OM'
        elif 'Om' in self.gdf.columns:
            return 'Om'
        elif 'om' in self.gdf.columns:
            return 'om'
        return None

    def calculate_formula(self, row):
        """
        [핵심] 비료 처방 공식 (사용자가 수정할 부분)
        :return: 10a당 필요 비료량 (kg/10a)
        """
        # 현재 공식: (OM * 2)
        # 만약 공식이 바뀌면 여기 숫자나 로직만 바꾸세요.
        om_value = row[self.om_col]
        n_need = om_value * 2

        # 음수 방지 (0보다 작으면 0)
        return max(n_need, 0)

    def interpolate_missing_data(self):
        """
        N_Need_10a 값이 0인 구간을 이웃 그리드 평균값으로 보정
        """
        # 0인 값 찾기
        zero_indices = self.gdf[self.gdf['N_Need_10a'] == 0].index

        if len(zero_indices) > 0:
            print(f"  [Calculator] 데이터 공백(0값) 보정 중... 대상: {len(zero_indices)}개 셀")

            for idx in zero_indices:
                current_geom = self.gdf.at[idx, 'geometry']
                # 맞닿은 이웃 찾기
                neighbors = self.gdf[self.gdf.geometry.touches(current_geom)]
                # 유효한(0보다 큰) 이웃 값만 추출
                valid_vals = neighbors[neighbors['N_Need_10a'] > 0]['N_Need_10a']

                if not valid_vals.empty:
                    # 평균값으로 대체
                    self.gdf.at[idx, 'N_Need_10a'] = valid_vals.mean()

    def execute(self):
        """
        전체 계산 프로세스 실행
        :return: 계산이 완료된 GeoDataFrame
        """
        if not self.om_col:
            print("  [Calculator] 주의: 'OM' 컬럼이 없어 계산을 생략하고 0으로 채웁니다.")
            self.gdf['N_Need_10a'] = 0
            self.gdf['N_Need_area'] = 0
            return self.gdf

        print(f"  [Calculator] '{self.om_col}' 컬럼 기반 비료량 산출 시작")

        # 1. 10a당 시비량(Rate) 계산
        self.gdf['N_Need_10a'] = self.gdf.apply(self.calculate_formula, axis=1)

        # 2. 0값 보정 (Interpolation)
        self.interpolate_missing_data()

        # 3. 면적 기반 총량(Mass) 계산
        # 그리드 면적 (m2)
        self.gdf['Grid_Area'] = self.gdf.geometry.area
        # 공식: Rate(kg/10a) * (면적 / 1000)
        self.gdf['N_Need_area'] = self.gdf['N_Need_10a'] * (self.gdf['Grid_Area'] / 1000)

        print("  [Calculator] 산출 완료 (Rate & Area)")

        return self.gdf