# Reinforcement Learning with Gymnasium (OpenAI Gym)

This guide walks you through setting up a reinforcement learning environment using Gymnasium (the maintained fork of OpenAI Gym) on Windows.

## Prerequisites

- Python 3.7+ installed on your system
- VS Code (recommended) or any other code editor
- PowerShell or Command Prompt
- Administrator rights on your system (for execution policy)

## Project Setup

1. **Create Project Directory**
   ```powershell
   # Create and navigate to project directory
   mkdir gym_setup
   cd gym_setup
   ```

2. **Set Up Virtual Environment**
   ```powershell
   # Create virtual environment
   python -m venv rl_env
   ```

3. **Activate Virtual Environment**
   
   If you encounter execution policy errors in PowerShell, you have three options:

   a. Option 1 - Run PowerShell as Administrator and execute:
   ```powershell
   Set-ExecutionPolicy RemoteSigned
   ```
   Then activate the environment:
   ```powershell
   .\rl_env\Scripts\activate
   ```

   b. Option 2 - Bypass policy for just this process:
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force; .\rl_env\Scripts\activate
   ```

   c. Option 3 - Use Command Prompt instead:
   ```cmd
   .\rl_env\Scripts\activate.bat
   ```

4. **Create Requirements File**
   
   Create a file named `requirements.txt` with the following content:
   ```txt
   gymnasium[classic-control]==0.29.1
   numpy>=1.23.1
   matplotlib>=3.7.1
   pygame>=2.5.0
   ```

5. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

## Testing the Environment

1. Create a file named `main.py` with the following content:
```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

def test_environment():
    """
    Test function to verify the environment setup is working correctly
    """
    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Keep track of episodes
    episode = 0
    total_reward = 0
    
    try:
        while True:  # Run indefinitely
            episode += 1
            observation, info = env.reset(seed=42)
            episode_reward = 0
            
            while True:  # Run until episode ends
                # Take a random action
                action = env.action_space.sample()
                
                # Step the environment
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                # Add a small delay to make the visualization easier to follow
                time.sleep(0.01)
                
                # If episode is done, print stats and start new episode
                if terminated or truncated:
                    print(f"Episode {episode} finished with reward: {episode_reward}")
                    break
    
    except KeyboardInterrupt:
        print("\nSimulation ended by user")
    finally:
        env.close()
        print("Environment closed successfully!")

if __name__ == "__main__":
    test_environment()
```

2. **Run the Test**
   ```powershell
   python main.py
   ```

## Project Structure
Your final project structure should look like this:
```
gym_setup/
├── rl_env/              # Virtual environment directory
├── requirements.txt     # Project dependencies
└── main.py             # Main Python script
```

## Understanding the CartPole Environment

- **Objective**: Keep the pole balanced upright on the cart
- **Actions**: 
  - 0: Push cart left
  - 1: Push cart right
- **Termination Conditions**:
  - Pole falls more than 15 degrees from vertical
  - Cart moves more than 2.4 units from center
  - Maximum episode length (500 steps) reached
- **Reward**: +1 for each timestep the pole remains balanced

## Additional Resources

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Reinforcement Learning Introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Gymnasium GitHub Repository](https://github.com/Farama-Foundation/Gymnasium)
