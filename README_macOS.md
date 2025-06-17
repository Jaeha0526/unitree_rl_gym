# Unitree RL Gym on macOS

This guide covers running Unitree RL Gym on macOS systems, including setup, capabilities, and limitations.

## 🚀 Quick Setup

### Prerequisites
- macOS (tested on Apple Silicon)
- Python 3.8+ 
- Virtual environment (recommended)

### Installation

1. **Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Run setup script:**
```bash
./setup_macos.sh
```

The setup script will automatically install all compatible dependencies and provide guidance on what works on macOS.

## 🎮 What You CAN Do on macOS

### ✅ Mujoco Simulation
Full physics simulation with pretrained models works perfectly:

#### G1 Robot
```bash
mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

#### H1 Robot
```bash
mjpython deploy/deploy_mujoco/deploy_mujoco.py h1.yaml
```

#### H1_2 Robot
```bash
mjpython deploy/deploy_mujoco/deploy_mujoco.py h1_2.yaml
```

### ✅ Model Analysis
- Load and inspect pretrained models
- Analyze policy networks
- Visualize training data (if available)

### ✅ Code Development
- Modify simulation parameters
- Develop new control algorithms
- Create custom environments (Mujoco-based)

## ❌ What You CANNOT Do on macOS

### Isaac Gym Training
- **Issue**: Isaac Gym requires Linux + NVIDIA GPU
- **Workaround**: Use Linux system or cloud instance for training
- **Commands that won't work**:
  ```bash
  python legged_gym/scripts/train.py --task=g1  # ❌ Fails
  python legged_gym/scripts/play.py --task=g1   # ❌ Fails
  ```

### Physical Robot Deployment
- **Issue**: Unitree SDK2 requires Linux for full functionality
- **Workaround**: Use Linux system connected to physical robot
- **Commands that may fail**:
  ```bash
  python deploy/deploy_real/deploy_real.py enp3s0 g1.yaml  # ❌ May fail
  ```

## 🔧 Available Pretrained Models

| Robot | Model Path | Mujoco Config | Status |
|-------|------------|---------------|---------|
| G1 | `deploy/pre_train/g1/motion.pt` | `g1.yaml` | ✅ Working |
| H1 | `deploy/pre_train/h1/motion.pt` | `h1.yaml` | ✅ Working |
| H1_2 | `deploy/pre_train/h1_2/motion.pt` | `h1_2.yaml` | ✅ Working |
| Go2 | ❌ Not available | ❌ Not available | ❌ Need to train |

## 🎯 Mujoco Simulation Details

### Controls
- The simulation runs automatically with the pretrained policy
- Robot will demonstrate walking/locomotion behaviors
- Simulation duration and parameters are configurable in YAML files

### Configuration Files
Located in `deploy/deploy_mujoco/configs/`:
- `g1.yaml` - G1 robot configuration
- `h1.yaml` - H1 robot configuration  
- `h1_2.yaml` - H1_2 robot configuration

### Key Parameters
- `simulation_duration`: How long to run (seconds)
- `simulation_dt`: Physics timestep
- `control_decimation`: Control frequency
- `policy_path`: Path to trained model
- `xml_path`: Robot model file

## 🔍 Troubleshooting

### Common Issues

1. **"launch_passive requires mjpython"**
   - Solution: Use `mjpython` instead of `python`
   - Correct: `mjpython deploy/deploy_mujoco/deploy_mujoco.py g1.yaml`

2. **Module not found errors**
   - Solution: Ensure virtual environment is activated
   - Solution: Run `./setup_macos.sh` to install dependencies

3. **Viewer not opening**
   - Solution: Ensure you have display access
   - Solution: Try running with `DISPLAY` environment variable

### Performance Tips
- Close other applications for better simulation performance
- Adjust `simulation_dt` in config files for smoother/faster simulation
- Use `control_decimation` to reduce control frequency if needed

## 🌍 Cross-Platform Workflow

### Recommended Setup
1. **Development**: macOS (this setup) for code development and Mujoco testing
2. **Training**: Linux system with NVIDIA GPU for Isaac Gym training
3. **Deployment**: Linux system connected to physical robot

### File Sharing
- Use git to sync code between systems
- Transfer trained models (`.pt` files) between systems
- Share configuration files for consistent behavior

## 📚 Next Steps

1. **Test the setup**: Run a Mujoco simulation
2. **Explore configs**: Modify YAML files to understand parameters
3. **For training**: Set up Linux environment with Isaac Gym
4. **For deployment**: Prepare Linux system with robot connection

## 🤝 Contributing

When developing on macOS:
- Test Mujoco simulations thoroughly
- Document any macOS-specific issues
- Ensure changes work across platforms
- Use Linux for full testing before submitting PRs

## 📄 See Also

- [Main README](README.md) - Complete project documentation
- [Setup Guide](doc/setup_en.md) - Full installation instructions
- [Physical Deployment](deploy/deploy_real/README.md) - Robot deployment guide