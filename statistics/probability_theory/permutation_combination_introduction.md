# What are Permutations and Combinations?
Permutations and combinations are mathematical techniques used to count and arrange objects without actually listing them.

## 1. Permutations
A permutation is an arrangement of all or part of a set of objects, with regard to the order of the arrangement. It tells us how many different ways we can arrange a set of items.

### Factorial Notation
- The factorial of a number \(n\) (denoted \(n!\)) is the product of all positive integers less than or equal to \(n\). It is the total number of ways to arrange \(n\) distinct objects into a sequence.
- \(n! = n \times (n-1) \times (n-2) \times \ldots \times 1\)

### Calculation of Permutations
- The number of permutations of \(n\) distinct objects taken \(r\) at a time is denoted as \(P(n, r)\) and calculated as:
  \[
  P(n, r) = \frac{n!}{(n-r)!}
  \]

### Example:
To find the number of ways to arrange 3 books out of a shelf of 5, calculate:
\[ P(5, 3) = \frac{5!}{(5-3)!} = \frac{5 \times 4 \times 3 \times 2 \times 1}{2 \times 1} = 60 \]

## 2. Combinations
A combination is a selection of all or part of a set of objects, without regard to the order of the objects. It tells us how many ways we can choose a subset of items from a larger set.

### Calculation of Combinations
- The number of combinations of \(n\) distinct objects taken \(r\) at a time is denoted as \(C(n, r)\) or \(\binom{n}{r}\), and calculated as:
  \[
  C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}
  \]

### Example:
To find the number of ways to select 3 books from a shelf of 5, calculate:
\[ C(5, 3) = \binom{5}{3} = \frac{5!}{3!(5-3)!} = \frac{5 \times 4 \times 3 \times 2 \times 1}{(3 \times 2 \times 1) \times (2 \times 1)} = 10 \]

# Applications of Permutations and Combinations
- **Cryptography**: Used in algorithms for arranging and selecting keys.
- **Game Theory and Gambling**: To calculate probabilities of various outcomes.
- **Computer Science**: In algorithms for searching and sorting.
- **Scheduling and Planning**: Determining possible schedules or plans.

# Examples with Python Code

## Calculating Factorials, Permutations, and Combinations with Python

Python’s `math` library provides functions to compute factorials, and you can define functions for permutations and combinations.

```python
import math

# Factorial
def factorial(n):
    return math.factorial(n)

# Permutation
def permutation(n, r):
    return math.factorial(n) // math.factorial(n - r)

# Combination
def combination(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Examples
print("Factorial of 5:", factorial(5))
print("Permutations of 5 taken 3 at a time:", permutation(5, 3))
print("Combinations of 5 taken 3 at a time:", combination(5, 3))
```
# Conclusion
Understanding permutations and combinations is essential for solving many problems in mathematics and statistics, especially those involving probability and arrangement. These tools help in making precise calculations without the need for exhaustive enumeration.
