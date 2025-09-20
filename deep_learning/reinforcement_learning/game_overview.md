# 🎮 Case Study: RL in Games (CartPole Example)

## 🕹️ The Game

* A cart 🛒 moves left/right on a track.
* A pole 🏗️ is balanced on top of it.
* **Goal:** Keep the pole from falling.
* **Reward:** +1 for every step the pole stays up.

👉 Why it’s perfect for beginners:

* Small, simple state (just numbers: position, velocity, angle).
* Two possible actions: **move left** ⬅️ or **move right** ➡️.

---

## 🧠 How RL Learns Here

1. **Agent starts dumb** 🤪 → moves randomly, pole falls fast.
2. **Tries actions** → gets **reward** (+1 per step).
3. Learns **patterns**:

   * “If pole leans right, move right.”
   * “If pole leans left, move left.”
4. Over many episodes, the **policy improves** until it balances for hundreds of steps.

---

## ⚡ Code Example (with Stable Baselines3)

You can run this in Python after installing `gymnasium` and `stable-baselines3`:

```python
import gymnasium as gym
from stable_baselines3 import PPO

# 1. Make the CartPole environment
env = gym.make("CartPole-v1")

# 2. Create an RL agent (PPO = Proximal Policy Optimization)
model = PPO("MlpPolicy", env, verbose=1)

# 3. Train the agent
model.learn(total_timesteps=10000)

# 4. Test the trained agent
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # shows the game (may need proper render setup)
```

👉 The agent will start clueless 😅, but after training, you’ll see it **balancing the pole like a pro 🤹**.

---

## 🕹️ Beyond CartPole → Bigger Games

Once you understand CartPole, you can try more exciting environments:

* **Atari games** (e.g., Breakout, Pong) — RL agents learn to hit the ball or score points.
* **Snake** — agent learns to grow without crashing.
* **Chess/Go** — more complex, but the same reward idea (win = +1, lose = -1).

For Atari in Gymnasium:

```python
env = gym.make("ALE/Breakout-v5", render_mode="human")
```

And you can train it with the same PPO/DQN agent — just give it more time and compute power ⚡.

---

# ✅ Why Games Are Great for Learning RL

* You **see learning happen** step by step 👀
* Rewards are natural (score, survival time, win/loss) ⭐
* Scales from simple toy games → world-class AI research
