# main.py
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def test_environment():
    """
    Test function to verify the environment setup is working correctly
    """
    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Reset the environment and get initial observation
    observation, info = env.reset(seed=42)
    
    # Run for 1000 timesteps
    for _ in range(1000):
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # If episode is done, reset the environment
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    print("Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()