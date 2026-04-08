#!/usr/bin/env python3
"""
Script to test if PyTorch is working correctly.
Tests basic PyTorch functionality, CUDA availability, and tensor operations.
"""

import sys


def test_pytorch():
    """Run PyTorch tests."""
    print("=" * 60)
    print("PyTorch Installation and Functionality Test")
    print("=" * 60)
    
    # Test 1: Import PyTorch
    print("\n1. Testing PyTorch import...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    # Test 2: Check CUDA availability
    print("\n2. Checking CUDA availability...")
    cuda_available = torch.cuda.is_available()
    print(f"✓ CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - Number of GPUs: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"  - GPU name: {torch.cuda.get_device_name(0)}")
    
    # Test 3: Create tensors on CPU
    print("\n3. Testing tensor creation on CPU...")
    try:
        x = torch.randn(3, 4)
        print(f"✓ Created CPU tensor with shape: {x.shape}")
    except Exception as e:
        print(f"✗ Failed to create tensor: {e}")
        return False
    
    # Test 4: Basic tensor operations
    print("\n4. Testing basic tensor operations...")
    try:
        y = torch.randn(3, 4)
        z = x + y
        w = torch.matmul(x, y.t())
        print(f"✓ Addition successful: {z.shape}")
        print(f"✓ Matrix multiplication successful: {w.shape}")
    except Exception as e:
        print(f"✗ Failed tensor operations: {e}")
        return False
    
    # Test 5: GPU tensor operations (if available)
    if cuda_available:
        print("\n5. Testing GPU tensor operations...")
        try:
            x_gpu = x.to("cuda")
            y_gpu = y.to("cuda")
            z_gpu = x_gpu + y_gpu
            print(f"✓ GPU tensor creation and addition successful")
            print(f"  - Device: {z_gpu.device}")
        except Exception as e:
            print(f"✗ Failed GPU operations: {e}")
            return False
    else:
        print("\n5. Skipping GPU tests (CUDA not available)")
    
    # Test 6: Gradients
    print("\n6. Testing autograd (gradients)...")
    try:
        x_grad = torch.randn(2, 3, requires_grad=True)
        y_grad = (x_grad ** 2).sum()
        y_grad.backward()
        print(f"✓ Autograd works correctly")
        print(f"  - Gradient shape: {x_grad.grad.shape}")
    except Exception as e:
        print(f"✗ Failed autograd test: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed! PyTorch is working correctly.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_pytorch()
    sys.exit(0 if success else 1)
