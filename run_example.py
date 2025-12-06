"""
전체 파이프라인 실행 예시 스크립트
데이터 생성부터 모델 학습까지 한 번에 실행합니다.
"""
import subprocess
import sys
import os

def run_command(command, description):
    """명령어 실행 헬퍼 함수"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"실행 명령: {command}")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ 성공!")
        if result.stdout:
            print(result.stdout)
    else:
        print("✗ 실패!")
        if result.stderr:
            print(result.stderr)
        return False
    
    return True

def main():
    """메인 실행 함수"""
    print("도쿄 23구 집값 예측 AI 프로젝트 실행")
    print("="*60)
    
    # 1. 데이터 생성
    if not run_command(
        "python generate_data.py",
        "1단계: 데이터 생성"
    ):
        print("데이터 생성 실패. 종료합니다.")
        return
    
    # 사용자에게 어떤 모델을 실행할지 선택하게 함
    print("\n" + "="*60)
    print("어떤 모델을 학습하시겠습니까?")
    print("1. AutoML 모델 (AutoKeras)")
    print("2. 커스텀 딥러닝 모델")
    print("3. 둘 다 실행")
    print("="*60)
    
    choice = input("선택 (1/2/3, 기본값: 3): ").strip() or "3"
    
    if choice == "1" or choice == "3":
        # 2. AutoML 모델 학습
        run_command(
            "python automl_model.py",
            "2단계: AutoML 모델 학습"
        )
    
    if choice == "2" or choice == "3":
        # 3. 커스텀 딥러닝 모델 학습
        run_command(
            "python custom_deep_learning_model.py",
            "3단계: 커스텀 딥러닝 모델 학습"
        )
    
    print("\n" + "="*60)
    print("모든 작업이 완료되었습니다!")
    print("="*60)

if __name__ == '__main__':
    main()
