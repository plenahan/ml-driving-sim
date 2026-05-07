import subprocess
import os

def run_ppo_experiments(num_runs=10):
    print(f"Starting {num_runs} runs for PPO...")
    
    # Open the log file in append mode
    with open("ppo_statistical_log.txt", "a") as log_file:
        for i in range(1, num_runs + 1):
            print(f"--- Running PPO Iteration {i}/{num_runs} ---")
            
            # Define dynamic paths to prevent overwriting
            actor_path = f"checkpoints/ppo_run_{i}/actor_{i}.pt"
            critic_path = f"checkpoints/ppo_run_{i}/critic_{i}.pt"
            plot_dir = f"checkpoints/ppo_run_{i}/plots"
            
            # Build the terminal command as a list
            command = [
                "python", "main.py", "train", 
                "--steps", "2000000", 
                "--actor-path", actor_path, 
                "--critic-path", critic_path, 
                "--plot-dir", plot_dir
            ]
            
            # Execute the command and pipe all print statements into the text file
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            
    print("All PPO runs complete!\n")


def run_sac_experiments(num_runs=10):
    print(f"Starting {num_runs} runs for SAC...")
    
    # Replace 'sac_main.py' with whatever you named your SAC execution file
    sac_script_name = "sac\main.py" 
    
    with open("sac_statistical_log.txt", "a") as log_file:
        for i in range(1, num_runs + 1):
            print(f"--- Running SAC Iteration {i}/{num_runs} ---")
            
            command = ["python", sac_script_name]
            
            # Execute the command and pipe output to the text file
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            
    print("All SAC runs complete!")


if __name__ == "__main__":
    # You can comment one of these out if you only want to run one algorithm at a time
    run_ppo_experiments(num_runs=2)
    run_sac_experiments(num_runs=2)