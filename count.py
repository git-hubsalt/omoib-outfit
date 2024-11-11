import os

# 확인할 폴더 경로 설정
folder_path = 'data/kfashion/image'  # 파일이 있는 폴더 경로

# 폴더 내 '.jpg' 파일 개수 세기
jpg_count = len([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
png_count = len([file for file in os.listdir(folder_path) if file.endswith('.png')])

print(f"폴더 내 '.jpg' 파일 개수: {jpg_count}")
print(f"폴더 내 '.png' 파일 개수: {png_count}")
