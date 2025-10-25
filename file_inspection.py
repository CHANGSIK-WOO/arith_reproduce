from PIL import Image, ImageFile
import os
from pathlib import Path

# 손상된 이미지 감지를 위해 False로 설정
ImageFile.LOAD_TRUNCATED_IMAGES = False


def find_corrupted_images(folder_path):
    """
    특정 폴더에서 손상된 이미지 파일을 찾는 함수

    Args:
        folder_path: 검사할 폴더 경로

    Returns:
        corrupted_images: 손상된 이미지 파일 경로 리스트
        total_images: 전체 이미지 파일 개수
    """
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

    corrupted_images = []
    total_images = 0

    print(f"📁 검사 중인 폴더: {folder_path}\n")

    # 폴더 내 모든 파일 검사
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 이미지 파일인지 확인
            if Path(file).suffix.lower() in image_extensions:
                total_images += 1
                file_path = os.path.join(root, file)

                try:
                    # 이미지 열기 시도
                    img = Image.open(file_path)
                    # 실제로 로드해보기 (여기서 손상 감지됨)
                    img.load()
                    img.close()
                    print(f"✅ {file}")

                except Exception as e:
                    # 손상된 이미지 발견!
                    print(f"❌ {file} - 에러: {str(e)[:50]}")
                    corrupted_images.append(file_path)

    return corrupted_images, total_images


# 실행 부분
if __name__ == "__main__":
    # 검사할 폴더 경로 입력
    folder_path = "/data/datasets/domainbed"

    # 경로가 존재하는지 확인
    if not os.path.exists(folder_path):
        print("❌ 경로가 존재하지 않습니다!")
    else:
        # 손상된 이미지 찾기
        corrupted_list, total = find_corrupted_images(folder_path)

        # 결과 출력
        print("\n" + "=" * 60)
        print("📊 검사 결과")
        print("=" * 60)
        print(f"전체 이미지 파일 개수: {total}개")
        print(f"손상된 이미지 개수: {len(corrupted_list)}개")
        print(f"정상 이미지 개수: {total - len(corrupted_list)}개")

        if corrupted_list:
            print("\n🔍 손상된 이미지 목록:")
            for idx, img_path in enumerate(corrupted_list, 1):
                print(f"  {idx}. {img_path}")
        else:
            print("\n✨ 모든 이미지가 정상입니다!")