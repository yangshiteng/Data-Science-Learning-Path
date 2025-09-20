# 🛠️ Setup (One-Time)

1. Install Python packages in your environment:

```bash
pip install "gymnasium[classic_control]" stable-baselines3 matplotlib imageio[ffmpeg]
```

2. In VS Code, open your Jupyter notebook (`.ipynb`) and select the same Python environment.

---

# 📒 Notebook Workflow

## 1. Imports

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import imageio
```

---

## 2. Make the Environment

```python
env = gym.make("CartPole-v1", render_mode="rgb_array")  # use rgb_array for Jupyter
```

---

## 3. Train an RL Agent

```python
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)
```

---

## 4. Evaluate the Agent

```python
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

---

## 5. Watch the Agent (Inline Video 🎥)

```python
frames = []
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    frames.append(env.render())

# Save video
video_path = "cartpole_demo.mp4"
imageio.mimsave(video_path, frames, fps=30)
video_path
```

➡️ In VS Code Jupyter, you can click the output link to play the MP4.

---

## 6. Plot Training Rewards (optional)

If you want to track rewards per episode:

```python
rewards = []
for i in range(10):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    rewards.append(total_reward)

plt.plot(rewards, marker='o')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Episodes")
plt.show()
```

---

# ✅ What You’ll See

* Training logs in the notebook output.
* A saved video (`cartpole_demo.mp4`) showing your agent balancing the pole 🤹.
* A reward curve plot 📈 showing how the agent performs across episodes.

---

⚡ Pro Tip: Once this works, you can replace `CartPole-v1` with other environments like `"MountainCar-v0"` or `"ALE/Pong-v5"` for more fun.
