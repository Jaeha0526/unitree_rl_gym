# Go2 Robot Deployment with MuJoCo

This README documents the process of deploying a pretrained Go2 robot policy in MuJoCo simulation, including the challenges encountered and solutions implemented.

The pretrained policy is based on the "Walk These Ways" framework adapted for Go2 robot, sourced from: https://github.com/Teddy-Liao/walk-these-ways-go2

## 1. Pretrained Files and Setup

### Model Files Location
The pretrained Go2 policy consists of two PyTorch JIT models located at:
```
deploy/pre_train/go2/
├── body_latest.jit           # Main policy network
└── adaptation_module_latest.jit  # Terrain adaptation module
```

**Source**: These models are from the "Walk These Ways" Go2 adaptation project (https://github.com/Teddy-Liao/walk-these-ways-go2), which implements a gait-conditioned reinforcement learning approach for robust quadrupedal locomotion.

### Two-Stage Policy Architecture
The Go2 uses a two-stage neural network architecture:

1. **Adaptation Module**: Processes observation history to extract latent features for terrain adaptation
   - Input: 2100 dimensions (30 timesteps × 70 dims per timestep)
   - Output: Latent vector for terrain/environment adaptation

2. **Body Network**: Main policy that generates motor commands
   - Input: Concatenation of observation history (2100 dims) + latent vector
   - Output: 12-dimensional action vector (joint position targets)

### XML Model Files
Created MuJoCo XML files for Go2 robot:
```
resources/robots/go2/
├── scene.xml    # Scene setup with lighting and ground
└── go2.xml      # Go2 robot model with 12 DOF kinematics
```

## 2. Observation and Action Structure

### Observation Structure (70 dimensions per timestep)
Each timestep contains exactly 70 dimensions:

```python
# 1. Projected gravity (3 dims)
gravity_vec = get_gravity_orientation(quaternion)

# 2. Gait commands (15 dims)
commands = [lin_vel_x, lin_vel_y, ang_vel_yaw, body_height, 
           gait_frequency, gait_phase, gait_offset, gait_bound,
           gait_duration, footswing_height, body_pitch, body_roll,
           aux_reward_coef, compliance, stance_width]

# 3. Joint positions relative to default (12 dims)
joint_positions = (current_positions - default_angles) * position_scale

# 4. Joint velocities (12 dims)  
joint_velocities = current_velocities * velocity_scale

# 5. Current actions (12 dims)
current_actions = action

# 6. Previous actions (12 dims)
previous_actions = prev_action

# 7. Clock inputs (4 dims)
clock = [sin(2πφ), cos(2πφ), sin(4πφ), cos(4πφ)]
```

**Total: 3 + 15 + 12 + 12 + 12 + 12 + 4 = 70 dimensions**

### Gait Commands Explanation (15 dimensions)

The gait commands are the key high-level control inputs that tell the robot how to move. These 15-dimensional commands control various aspects of locomotion:

#### Velocity Commands (3 dims)
- **lin_vel_x**: Desired forward/backward velocity (m/s)
- **lin_vel_y**: Desired left/right strafe velocity (m/s)  
- **ang_vel_yaw**: Desired turning velocity (rad/s)

#### Body Pose (3 dims)
- **body_height**: Desired body height offset from nominal (m)
- **body_pitch**: Desired body pitch angle (rad)
- **body_roll**: Desired body roll angle (rad)

#### Gait Parameters (6 dims)
- **gait_frequency**: Step frequency in Hz (e.g., 2.5 Hz = 2.5 steps/second)
- **gait_phase**: Current phase in gait cycle (0.0 to 1.0)
- **gait_offset**: Phase offset between different legs
- **gait_bound**: Bounding parameter for gait
- **gait_duration**: Stance phase duration as fraction of gait cycle (0.5 = 50% stance, 50% swing)
- **footswing_height**: Maximum foot lift height during swing phase (m)

#### Style Parameters (3 dims)
- **stance_width**: Desired stance width (m)
- **compliance**: Joint compliance/stiffness parameter
- **aux_reward_coef**: Auxiliary reward coefficient for training

#### Progressive Gait Command Strategy

Our simulation uses a progressive approach to gait commands:

```python
def get_gait_commands(time_elapsed):
    if time_elapsed < 5.0:
        # Phase 1: Standing stability test
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.25]
        
    elif time_elapsed < 15.0:
        # Phase 2: Early forward walk
        return [0.3, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.0, 0.0, 0.25]
        
    else:
        # Phase 3: Steady walking
        return [0.4, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.5, 0.12, 0.0, 0.0, 0.0, 0.0, 0.25]
```

**Key Insights**:
- **Conservative frequencies**: We use 2.5 Hz instead of higher frequencies for stability
- **Moderate footswing**: 0.12-0.15m lift height prevents excessive leg lifting
- **Consistent forward velocity**: 0.3-0.4 m/s provides steady forward motion
- **50% stance duration**: Balanced between stability and agility

### Clock Inputs Explanation (4 dimensions)

Clock inputs provide periodic timing signals that help the policy coordinate leg movements in a gait pattern:

```python
def get_clock_inputs(phase):
    """Generate 4-dimensional gait phase signals"""
    clock = np.zeros(4)
    clock[0] = np.sin(2 * np.pi * phase)    # Fundamental frequency
    clock[1] = np.cos(2 * np.pi * phase)    # Fundamental frequency (90° offset)
    clock[2] = np.sin(4 * np.pi * phase)    # Double frequency
    clock[3] = np.cos(4 * np.pi * phase)    # Double frequency (90° offset)
    return clock
```

**Purpose**:
- **Phase tracking**: Helps policy know where in the gait cycle each leg should be
- **Coordination**: Enables proper timing between different legs (e.g., trot gait)
- **Smooth transitions**: Sin/cos pairs provide smooth, continuous timing signals
- **Multi-frequency**: Fundamental + double frequency allows complex gait patterns

**Gait Phase Calculation**:
```python
gait_period = 0.5  # seconds (for 2 Hz gait)
gait_phase = (time_elapsed % gait_period) / gait_period  # 0.0 to 1.0
```

The clock inputs are essential for quadrupedal locomotion as they provide the temporal structure needed for coordinated leg movements.

### History Buffer
- **Length**: 30 timesteps
- **Total input**: 30 × 70 = 2100 dimensions
- **Buffer type**: Rolling deque that maintains recent observation history

### Action Structure (12 dimensions)
Actions represent target joint positions for the 12 actuated joints:

```python
# Joint mapping (Front-Left, Front-Right, Rear-Left, Rear-Right)
joints = [
    'FL_hip_joint',   'FL_thigh_joint',   'FL_calf_joint',    # 0,1,2
    'FR_hip_joint',   'FR_thigh_joint',   'FR_calf_joint',    # 3,4,5  
    'RL_hip_joint',   'RL_thigh_joint',   'RL_calf_joint',    # 6,7,8
    'RR_hip_joint',   'RR_thigh_joint',   'RR_calf_joint'     # 9,10,11
]

# Action scaling
scaled_action = action * action_scale  # action_scale = 0.25
scaled_action[hip_indices] *= 0.5      # Additional scaling for hip joints

# Convert to target positions
target_positions = scaled_action + default_angles
```

## 3. Simulation Code Explanation

### Main Script: `deploy_mujoco_go2_correct.py`

#### Key Components

1. **Model Loading**
```python
adaptation_module = torch.jit.load(adaptation_path)
body = torch.jit.load(body_path)
```

2. **Observation History Management**
```python
obs_history = deque(maxlen=history_length)  # 30 timesteps
obs_history.append(current_obs)             # Add new observation
obs_history_array = np.array(obs_history).flatten()  # Convert to 2100-dim vector
```

3. **Two-Stage Inference**
```python
with torch.no_grad():
    # Stage 1: Extract terrain adaptation features
    latent = adaptation_module.forward(obs_history_tensor)
    
    # Stage 2: Generate actions
    body_input = torch.cat((obs_history_tensor, latent), dim=-1)
    action_tensor = body.forward(body_input)
```

4. **PD Control**
```python
tau = pd_control(target_dof_pos, current_pos, kps, target_vel, current_vel, kds)
```

### Running the Simulation

#### Prerequisites
- MuJoCo installation with mjpython support
- PyTorch with JIT model support
- Go2 model files in correct locations

#### Command
```bash
mjpython deploy/deploy_mujoco/deploy_mujoco_go2_correct.py go2.yaml
```

#### Configuration (go2.yaml)
```yaml
xml_path: "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/scene.xml"
simulation_duration: 10.0
simulation_dt: 0.002
control_decimation: 10  # 50Hz control frequency

# PD Controller gains
kps: [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
kds: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]

# Default joint angles (fixed asymmetry issue)
default_angles: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]
```

## 4. Problems Found and Debugging

### Problem 1: Joint Asymmetry Causing Spinning
**Issue**: Robot spun in circles instead of walking forward.

**Root Cause**: Asymmetric default joint angles - rear thigh joints at 1.0 rad vs front thigh joints at 0.8 rad.

**Solution**: 
```yaml
# Before (causing spinning)
default_angles: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]

# After (symmetric)  
default_angles: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
                 0.1, 0.8, -1.5, -0.1, 0.8, -1.5]
```

### Problem 2: Robot Gets "Stuck" in Shivering State
**Issue**: Robot walks initially but then gets stuck in a state where it just shivers without moving forward.

**Symptoms**:
- Actions continue changing but robot velocity drops to near zero
- Robot exhibits small oscillatory movements without forward progress
- Behavior persists indefinitely without intervention

**Debugging Implementation**:
```python
# 1. Velocity monitoring
robot_velocity_magnitude = np.sqrt(d.qvel[0]**2 + d.qvel[1]**2)
is_low_velocity = robot_velocity_magnitude < 0.1 and time_elapsed > 10.0

# 2. Action repetition detection  
action_change = np.abs(action - prev_action).max()
is_repetitive_actions = action_change < 0.3

# 3. Stuck detection
if is_low_velocity or is_repetitive_actions:
    stuck_counter += 1
    if stuck_counter >= 3:  # 1.5 seconds of stuck behavior
        reset_observation_history()
        stuck_counter = 0
```

**Solution Attempts**:

1. **Observation History Reset**: Clear the 30-timestep history buffer when stuck behavior is detected
2. **Conservative Gait Commands**: Reduce gait frequency and footswing height for stability
3. **Sensitive Stuck Detection**: Monitor both velocity and action variation
4. **Gait Phase Reset**: Reset walking cycle timing along with observation history

### Problem 3: Observation Dimension Mismatch
**Issue**: Initial confusion about observation structure - policy expected 70 dims per timestep, not terrain heights.

**Resolution**: Correct observation structure implementation:
- No terrain height measurements in observations
- Exactly 70 dimensions per timestep as specified
- Proper scaling and normalization of all observation components

### Current Status and Remaining Issues

**Working Components**:
- ✅ Correct observation structure (70 dims × 30 timesteps)
- ✅ Two-stage policy inference
- ✅ Joint symmetry fixed
- ✅ Stuck detection and recovery mechanism
- ✅ Stable simulation environment

**Remaining Challenges**:
- 🔄 Robot still tends to get stuck in shivering state periodically
- 🔄 Reset mechanism provides temporary relief but doesn't prevent recurrence
- 🔄 May need policy retraining or different gait command strategies

### Debug Output Example
```
Time: 35.0s
Gait commands: [vel=0.30, freq=3.0, height=0.15]
Max action change: 2.303, Max action magnitude: 3.274
⚠️  WARNING: Robot stuck - vel_mag: 0.081, action_change: 2.303, count: 1
Robot position: x=-1.023, y=0.707, z=0.230
Robot velocity: vx=-0.065, vy=-0.048, vz=-0.013

Time: 40.0s  
⚠️  WARNING: Robot stuck - vel_mag: 0.064, action_change: 1.925, count: 2

Time: 45.0s
⚠️  WARNING: Robot stuck - vel_mag: 0.033, action_change: 2.318, count: 3
🔄 Resetting observation history due to stuck behavior
```

This demonstrates the stuck detection working correctly and the automatic recovery mechanism triggering when needed.