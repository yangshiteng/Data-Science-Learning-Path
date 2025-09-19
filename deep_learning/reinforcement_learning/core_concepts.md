# 🔹 Core Concepts of Reinforcement Learning

## 1. **Agent**

* The learner or decision-maker.
* Example: A robot, a game character, or even a recommendation system.

---

## 2. **Environment**

* The world the agent interacts with.
* Example: The game board, a simulation, or the real world.

---

## 3. **State (S)**

* A snapshot of the environment at a given time.
* Example:

  * In CartPole: position of the cart + angle of the pole.
  * In chess: current arrangement of pieces.

---

## 4. **Action (A)**

* A choice the agent makes.
* Example:

  * In CartPole: move left or right.
  * In chess: move a pawn, rook, etc.

---

## 5. **Reward (R)**

* Feedback from the environment (positive or negative).
* Guides the agent’s learning.
* Example:

  * +1 for balancing the pole longer.
  * –1 if the robot crashes.

---

## 6. **Policy (π)**

* The agent’s “strategy” or decision rule.
* Maps states → actions.
* Example: “If the pole leans right, move cart right.”

---

## 7. **Value (V)**

* How good a state is (expected long-term reward).
* Example: In chess, a board position with strong advantage has high value.

---

## 8. **Q-Value (Action-Value)**

* How good it is to take a specific action in a specific state.
* Example: “If I move left now, I expect +5 reward in the future.”

---

## 9. **Exploration vs. Exploitation**

* **Exploration:** Try new actions to discover better rewards.
* **Exploitation:** Stick with known actions that give high reward.
* Example:

  * Try a new move in chess (explore).
  * Keep using a winning strategy (exploit).

---

## 10. **Episode**

* A complete sequence of interaction from start → finish.
* Example:

  * One full game of chess.
  * One run of CartPole until the pole falls.

---

✨ **Simple analogy:**

* Imagine training a dog 🐶.

  * **State:** Dog is sitting.
  * **Action:** You say “roll over.”
  * **Reward:** Dog gets a treat if it does well.
  * **Policy:** Dog learns which commands = treats.
