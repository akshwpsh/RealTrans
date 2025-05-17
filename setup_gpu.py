#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CUDA 지원 PyTorch 설치 및 GPU 사용 가능 여부 확인 스크립트
"""

import os
import sys
import subprocess
import platform

def check_cuda_available():
    """CUDA 설치 여부 확인"""
    try:
        # nvcc 버전 확인
        nvcc_info = subprocess.check_output('nvcc --version', shell=True).decode('utf-8')
        print(f"CUDA 컴파일러 정보:\n{nvcc_info}")
        return True
    except:
        print("CUDA가 설치되어 있지 않거나 PATH에 포함되어 있지 않습니다.")
        return False

def check_pytorch_cuda():
    """PyTorch CUDA 지원 확인"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "없음"
            
            print(f"\nPyTorch CUDA 정보:")
            print(f"- PyTorch 버전: {torch.__version__}")
            print(f"- CUDA 버전: {cuda_version}")
            print(f"- GPU 장치 수: {device_count}")
            print(f"- GPU 장치명: {device_name}")
            print(f"- CUDA 사용 가능: {cuda_available}")
            return True
        else:
            print("\nPyTorch가 설치되어 있지만 CUDA를 사용할 수 없습니다.")
            print(f"- PyTorch 버전: {torch.__version__}")
            print(f"- CUDA 사용 가능: {cuda_available}")
            return False
    except ImportError:
        print("\nPyTorch가 설치되어 있지 않습니다.")
        return False

def install_pytorch_cuda():
    """CUDA 지원 PyTorch 설치"""
    print("\nCUDA 지원 PyTorch 설치 옵션:")
    print("1. CUDA 12.8용 PyTorch 2.7.0 설치 (권장)")
    print("2. CUDA 12.1용 PyTorch 2.1.0 설치")
    print("3. CUDA 11.8용 PyTorch 2.1.0 설치")
    print("4. CPU 전용 PyTorch 설치")
    print("q. 종료")
    
    choice = input("\n선택: ").strip().lower()
    
    if choice == "1":
        cmd = "pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 --extra-index-url https://download.pytorch.org/whl/cu128"
        print(f"\n명령 실행: {cmd}")
        os.system(cmd)
    elif choice == "2":
        cmd = "pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121"
        print(f"\n명령 실행: {cmd}")
        os.system(cmd)
    elif choice == "3":
        cmd = "pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118"
        print(f"\n명령 실행: {cmd}")
        os.system(cmd)
    elif choice == "4":
        cmd = "pip install torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0"
        print(f"\n명령 실행: {cmd}")
        os.system(cmd)
    elif choice == "q":
        return
    else:
        print("잘못된 선택입니다.")
        return

def main():
    """메인 함수"""
    print("=" * 60)
    print(" RealTrans - CUDA 지원 설정 도우미")
    print("=" * 60)
    
    # 시스템 정보 표시
    print(f"운영체제: {platform.system()} {platform.release()}")
    
    # CUDA 확인
    cuda_installed = check_cuda_available()
    
    # PyTorch CUDA 지원 확인
    pytorch_cuda = check_pytorch_cuda()
    
    if not cuda_installed:
        print("\nCUDA를 설치하려면:")
        print("1. NVIDIA 웹사이트(https://developer.nvidia.com/cuda-downloads)에서 CUDA Toolkit을 다운로드하세요.")
        print("   - 권장 버전: CUDA 12.8 이상")
        print("2. 설치 후 컴퓨터를 재시작하세요.")
    
    if not pytorch_cuda:
        print("\nCUDA 지원 PyTorch를 설치하시겠습니까?")
        if input("설치 진행 (y/n): ").strip().lower() == "y":
            install_pytorch_cuda()
    
    print("\n설정 완료! 'python main.py'로 애플리케이션을 실행하세요.")

if __name__ == "__main__":
    main() 