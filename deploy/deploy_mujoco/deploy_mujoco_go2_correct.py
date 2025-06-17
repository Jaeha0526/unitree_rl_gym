import time
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from collections import deque


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def get_gait_commands(time_elapsed):
    """Generate 15-dimensional gait parameter commands with progressive testing"""
    commands = np.zeros(15)
    
    # Progressive command sequence based on time
    if time_elapsed < 5.0:
        # Phase 1 (0-5s): Standing stability test
        commands[0] = 0.0      # lin_vel_x (stationary)
        commands[1] = 0.0      # lin_vel_y  
        commands[2] = 0.0      # ang_vel_yaw
        commands[3] = 0.0      # body_height (nominal)
        commands[4] = 0.0      # gait_frequency (no gait)
        commands[5] = 0.0      # gait_phase
        commands[6] = 0.0      # gait_offset  
        commands[7] = 0.0      # gait_bound
        commands[8] = 0.5      # gait_duration (50% stance)
        commands[9] = 0.1      # footswing_height (10cm)
        commands[10] = 0.0     # body_pitch
        commands[11] = 0.0     # body_roll
        commands[12] = 0.0     # aux_reward_coef
        commands[13] = 0.0     # compliance
        commands[14] = 0.25    # stance_width
        if int(time_elapsed) % 2 == 0 and time_elapsed - int(time_elapsed) < 0.1:
            print(f"Phase 1: Standing stability test")
        
    elif time_elapsed < 15.0:
        # Phase 2 (5-15s): Start forward motion early
        commands[0] = 0.3      # lin_vel_x (slow forward motion)
        commands[1] = 0.0      # lin_vel_y  
        commands[2] = 0.0      # ang_vel_yaw
        commands[3] = 0.0      # body_height
        commands[4] = 2.5      # gait_frequency (slower, more stable)
        commands[5] = 0.0      # gait_phase
        commands[6] = 0.0      # gait_offset  
        commands[7] = 0.0      # gait_bound
        commands[8] = 0.5      # gait_duration
        commands[9] = 0.15     # footswing_height (lower for stability)
        commands[10] = 0.0     # body_pitch
        commands[11] = 0.0     # body_roll
        commands[12] = 0.0     # aux_reward_coef
        commands[13] = 0.0     # compliance
        commands[14] = 0.25    # stance_width
        if int(time_elapsed) % 2 == 0 and time_elapsed - int(time_elapsed) < 0.1:
            print(f"Phase 2: Early forward walk (vel=0.3, freq=2.5, height=0.15)")
        
    else:
        # Phase 4 (15s+): Steady forward walking - keep it simple
        commands[0] = 0.4      # lin_vel_x (consistent forward speed)
        commands[1] = 0.0      # lin_vel_y  
        commands[2] = 0.0      # ang_vel_yaw (no turning for now)
        commands[3] = 0.0      # body_height
        commands[4] = 2.5      # gait_frequency (consistent, stable frequency)
        commands[5] = 0.0      # gait_phase
        commands[6] = 0.0      # gait_offset  
        commands[7] = 0.0      # gait_bound
        commands[8] = 0.5      # gait_duration
        commands[9] = 0.12     # footswing_height (conservative for stability)
        commands[10] = 0.0     # body_pitch
        commands[11] = 0.0     # body_roll
        commands[12] = 0.0     # aux_reward_coef
        commands[13] = 0.0     # compliance
        commands[14] = 0.25    # stance_width
        if int(time_elapsed) % 5 == 0 and time_elapsed - int(time_elapsed) < 0.1:
            print(f"Phase 3: Steady walking (vel=0.4, freq=2.5, height=0.12)")
    
    return commands


def get_clock_inputs(phase):
    """Generate 4-dimensional gait phase signals"""
    clock = np.zeros(4)
    clock[0] = np.sin(2 * np.pi * phase)
    clock[1] = np.cos(2 * np.pi * phase)
    clock[2] = np.sin(4 * np.pi * phase)
    clock[3] = np.cos(4 * np.pi * phase)
    return clock


if __name__ == "__main__":
    # Load config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        
        # Observation scaling as per config
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05
        action_scale = 0.25
        hip_scale_reduction = 0.5

    # Initialize
    action = np.zeros(12, dtype=np.float32)
    prev_action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    
    # History buffer for 30 timesteps of 70-dim observations
    history_length = 30
    obs_per_timestep = 70
    obs_history = deque(maxlen=history_length)
    
    # Initialize history with zeros
    def reset_observation_history():
        obs_history.clear()
        for _ in range(history_length):
            obs_history.append(np.zeros(obs_per_timestep, dtype=np.float32))
    
    reset_observation_history()
    
    # Divergence detection
    stuck_counter = 0
    last_action_change = 0.0

    counter = 0
    gait_phase = 0.0
    gait_period = 0.5  # seconds

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load models
    print("Loading Go2 policy with correct observation structure...")
    body_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2/body_latest.jit"
    adaptation_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2/adaptation_module_latest.jit"
    
    adaptation_module = torch.jit.load(adaptation_path)
    body = torch.jit.load(body_path)
    print("Models loaded successfully!")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # Apply control
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Update time and gait phase
                time_elapsed = counter * simulation_dt * control_decimation
                gait_phase = (time_elapsed % gait_period) / gait_period
                
                # Get robot state
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                
                # Build observation with CORRECT 70-dim structure
                current_obs = np.zeros(obs_per_timestep, dtype=np.float32)
                idx = 0
                
                # 1. Projected gravity (3)
                gravity_vec = get_gravity_orientation(quat)
                current_obs[idx:idx+3] = gravity_vec
                idx += 3
                
                # 2. Commands - gait parameters (15)
                gait_commands = get_gait_commands(time_elapsed)
                current_obs[idx:idx+15] = gait_commands
                idx += 15
                
                # 3. Joint positions relative to default (12)
                qj_normalized = (qj - default_angles) * dof_pos_scale
                current_obs[idx:idx+12] = qj_normalized
                idx += 12
                
                # 4. Joint velocities (12)
                dqj_normalized = dqj * dof_vel_scale
                current_obs[idx:idx+12] = dqj_normalized
                idx += 12
                
                # 5. Current actions (12)
                current_obs[idx:idx+12] = action
                idx += 12
                
                # 6. Previous actions (12)
                current_obs[idx:idx+12] = prev_action
                idx += 12
                
                # 7. Clock inputs (4)
                clock = get_clock_inputs(gait_phase)
                current_obs[idx:idx+4] = clock
                idx += 4
                
                # Total: 3 + 15 + 12 + 12 + 12 + 12 + 4 = 70 dims exactly
                
                # Clip observations to ±100 as per config
                current_obs = np.clip(current_obs, -100, 100)
                
                # Add to history
                obs_history.append(current_obs)
                
                # Convert history to tensor (30 x 70 = 2100 dims)
                obs_history_array = np.array(obs_history).flatten()
                obs_history_tensor = torch.from_numpy(obs_history_array).unsqueeze(0).float()
                
                # Two-stage inference
                with torch.no_grad():
                    # Stage 1: Adaptation module
                    latent = adaptation_module.forward(obs_history_tensor)
                    
                    # Stage 2: Body network
                    body_input = torch.cat((obs_history_tensor, latent), dim=-1)
                    action_tensor = body.forward(body_input)
                    
                    # Update action history
                    prev_action = action.copy()
                    action = action_tensor.numpy().squeeze()
                
                # Apply correct action scaling
                scaled_action = action * action_scale
                
                # Hip joints get additional 0.5 scaling (indices 0,3,6,9)
                hip_indices = [0, 3, 6, 9]
                for i in hip_indices:
                    scaled_action[i] *= hip_scale_reduction
                
                # Convert to target positions
                target_dof_pos = scaled_action + default_angles
                
                # Debug: Print actions and check for divergence  
                if counter % (control_decimation * 10) == 0:  # Every 0.2 seconds
                    print(f"\nTime: {time_elapsed:.1f}s")
                    print(f"Gait commands: [vel={gait_commands[0]:.2f}, freq={gait_commands[4]:.1f}, height={gait_commands[9]:.2f}]")
                    
                    # Check if actions are changing significantly
                    if counter > control_decimation * 50:  # After first iteration
                        action_change = np.abs(action - prev_action).max()
                        action_magnitude = np.abs(action).max()
                        print(f"Max action change: {action_change:.4f}, Max action magnitude: {action_magnitude:.4f}")
                        
                        # Detect if robot is "stuck" - low velocity or repetitive actions
                        robot_velocity_magnitude = np.sqrt(d.qvel[0]**2 + d.qvel[1]**2)
                        is_low_velocity = robot_velocity_magnitude < 0.1 and time_elapsed > 10.0  # After initial settling
                        is_repetitive_actions = action_change < 0.3  # More sensitive threshold
                        
                        if is_low_velocity or is_repetitive_actions:
                            stuck_counter += 1
                            print(f"⚠️  WARNING: Robot stuck - vel_mag: {robot_velocity_magnitude:.3f}, action_change: {action_change:.3f}, count: {stuck_counter}")
                            
                            # Reset observation history if stuck for too long
                            if stuck_counter >= 3:  # 1.5 seconds of being stuck
                                print("🔄 Resetting observation history due to stuck behavior")
                                reset_observation_history()
                                stuck_counter = 0
                                # Also reset gait phase to restart walking cycle
                                gait_phase = 0.0
                        else:
                            stuck_counter = 0  # Reset counter if robot is moving normally
                        
                        # Detect if actions are exploding
                        if action_magnitude > 10.0:
                            print("⚠️  WARNING: Actions exploding (high magnitude)")
                            print("🔄 Resetting observation history due to action explosion")
                            reset_observation_history()
                    
                    # Check observation health
                    obs_magnitude = np.abs(current_obs).max()
                    print(f"Max observation magnitude: {obs_magnitude:.4f}")
                    
                    if obs_magnitude > 50.0:
                        print("⚠️  WARNING: Observations may be diverging")
                    
                    print(f"Robot position: x={d.qpos[0]:.3f}, y={d.qpos[1]:.3f}, z={d.qpos[2]:.3f}")
                    print(f"Robot velocity: vx={d.qvel[0]:.3f}, vy={d.qvel[1]:.3f}, vz={d.qvel[2]:.3f}")
                    print()

            viewer.sync()
            
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)