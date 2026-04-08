#!/usr/bin/env python3
"""
Hello World example script.
This is a simple starting point for your project.
"""

import torch


def main():
    """Print a greeting and display PyTorch information."""
    print("=" * 60)
    print("Hello from Engineering AI!")
    print("=" * 60)
    print()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    print("Your development environment is ready!")
    print("Start editing this file or create new scripts in src/")
    print("=" * 60)


if __name__ == "__main__":
    main()
