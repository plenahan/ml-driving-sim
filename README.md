# ml-driving-sim

### References

[PPO medium article](https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7)

[PPO PyTorch Tutorial Series Part 1](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8)

### Run

Train headless and save checkpoints:

```bash
python main.py train --steps 2000000 --actor-path checkpoints/actor.pt --critic-path checkpoints/critic.pt
```

Train and save per-metric line charts over total timesteps:

```bash
python main.py train --steps 2000000 --actor-path checkpoints/actor.pt --critic-path checkpoints/critic.pt --plot-dir checkpoints/training_plots
```

Optional: show charts interactively while also saving them:

```bash
python main.py train --steps 2000000 --show-plots
```

Play with deterministic policy rendering the simulation:

```bash
python main.py play --actor-path checkpoints/actor.pt --critic-path checkpoints/critic.pt --max-steps 5000
```

