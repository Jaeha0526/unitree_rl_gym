#!/bin/bash

# Unitree RL Gym Setup Script for macOS
# This script sets up the environment for Unitree RL Gym on macOS with Apple Silicon

set -e  # Exit on any error

echo "🤖 Unitree RL Gym Setup for macOS"
echo "=================================="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "It's recommended to run this in a virtual environment"
    echo "Create one with: python3 -m venv venv && source venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Installing dependencies..."

# Upgrade pip first
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version for macOS)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio

# Clone and install rsl_rl
echo "🏃 Installing RSL-RL..."
if [ ! -d "rsl_rl" ]; then
    git clone https://github.com/leggedrobotics/rsl_rl.git
fi
cd rsl_rl
git checkout v1.0.2
pip install -e .
cd ..

# Install Mujoco and other dependencies
echo "🎯 Installing Mujoco and other dependencies..."
pip install mujoco==3.2.3
pip install matplotlib pyyaml tensorboard

# Try to install Unitree SDK2 (optional, may fail on macOS)
echo "📡 Installing Unitree SDK2 (optional)..."
if [ ! -d "unitree_sdk2_python" ]; then
    git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
fi
pip install -e ./unitree_sdk2_python || echo "⚠️  Unitree SDK2 installation failed (expected on macOS)"

# Install unitree_rl_gym
echo "🦴 Installing Unitree RL Gym..."
pip install -e . --no-deps

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎮 What you can do now:"
echo "  • Test Mujoco simulation: mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml"
echo "  • Available robots: G1, H1, H1_2"
echo ""
echo "⚠️  Limitations on macOS:"
echo "  • Isaac Gym not supported (requires Linux + NVIDIA GPU)"
echo "  • Physical robot deployment may need Linux system"
echo "  • Mujoco simulation works great!"
echo ""
echo "📚 Next steps:"
echo "  1. Try: mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml"
echo "  2. For training: Use a Linux system with NVIDIA GPU"
echo "  3. For physical deployment: Use Linux system connected to robot"