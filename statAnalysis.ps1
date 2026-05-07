for ($i=1; $i -le 20; $i++) {
    Write-Host "Starting PPO Run $i..."
    
    # Define dynamic paths for this specific run
    $actor = "checkpoints/run_$i/actor_$i.pt"
    $critic = "checkpoints/run_$i/critic_$i.pt"
    $plotDir = "checkpoints/run_$i/plots"
    
    # Run the headless training and append the terminal output to a log file
    python main.py train --steps 2000000 --actor-path $actor --critic-path $critic --plot-dir $plotDir >> ppo_statistical_log.txt
}
Write-Host "All 20 PPO runs complete!"