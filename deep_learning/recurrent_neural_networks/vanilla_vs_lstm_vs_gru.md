## 📊 **Comparison: Vanilla RNN vs LSTM vs GRU**

| Feature                           | **Vanilla RNN**           | **LSTM**                                                 | **GRU**                                              |
| --------------------------------- | ------------------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| **Introduced**                    | Early 1990s               | 1997 (Hochreiter & Schmidhuber)                          | 2014 (Cho et al.)                                    |
| **Purpose**                       | Basic sequential modeling | Capture long-term dependencies, solve vanishing gradient | Simplified LSTM, faster training                     |
| **Hidden State**                  | $h_t$                     | $h_t$ + Cell state $C_t$                                 | Single hidden state $h_t$                            |
| **Memory**                        | Short-term memory only    | Short-term + long-term memory                            | Combined memory in hidden state                      |
| **Gates**                         | None                      | 3: Forget, Input, Output                                 | 2: Update, Reset                                     |
| **Vanishing/Exploding Gradients** | Prone                     | Reduced with memory cell                                 | Reduced (but less than LSTM for very long sequences) |
| **Equations Complexity**          | Simple                    | Complex (6+ equations)                                   | Medium (4 equations)                                 |
| **Training Time**                 | Fast                      | Slower (more parameters)                                 | Faster than LSTM                                     |
| **Performance**                   | Poor on long sequences    | Excellent for long sequences                             | Competitive with LSTM, often similar performance     |
| **Use Cases**                     | Short texts, small data   | Translation, text generation, speech                     | Same as LSTM, more efficient for some tasks          |
| **Parameter Count**               | Few                       | High                                                     | Fewer than LSTM                                      |

---

## 🧠 Summary

* **Vanilla RNN**: Easy to implement but struggles with long-term dependencies due to vanishing gradients.
* **LSTM**: Powerful and robust for complex sequence tasks, but computationally heavy.
* **GRU**: A lighter alternative to LSTM with fewer gates and similar performance in most tasks.

---

### 🔧 Choose:

* Use **Vanilla RNN** only for experiments or educational purposes.
* Use **LSTM** when capturing long-term memory is crucial.
* Use **GRU** when you want **faster training** with **comparable accuracy** to LSTM.

