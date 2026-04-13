import numpy as np

# 1. bin 파일 경로 지정
bin_path = 'result/TEST2_new/TEST2_new_20mx20m/TASKDATA/GRD00000.bin'

# 2. 파일 읽기 (리틀엔디안 32비트 부호 있는 정수: '<i4')
# ISO 11783-10 표준의 I="2" 규격에 해당합니다.
data = np.fromfile(bin_path, dtype='<i4')

# 3. 데이터 확인
print(f"총 그리드(바둑판) 개수: {len(data)} 개")
print(f"처방량 데이터 (처음 1000개):")
print(data[:1000])

# 4. (선택) 2차원 지도로 복원하기
# TASKDATA.XML 파일의 <GRD> 태그에 있는 E(너비)와 F(높이) 값을 입력하면
# 실제 바둑판 모양의 2차원 배열로 볼 수 있습니다.
# width = 15  # XML의 E 값 입력
# height = 20 # XML의 F 값 입력
# grid_2d = data.reshape((height, width))
# print(grid_2d)