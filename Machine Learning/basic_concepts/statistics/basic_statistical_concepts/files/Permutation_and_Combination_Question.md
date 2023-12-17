# Permutation-based Questions

## Permutation with repeated items

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/9600a979-5f3c-4cc4-ae36-2eb12e23d238)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/7d930217-70ac-4120-98db-39f319b3b86d)

![image](https://github.com/yangshiteng/Data-Science-Learning-Path/assets/60442877/1dfab870-a67e-489a-973b-b5ad7e4e3b83)



Question 1:

Google: Two teams play a series of games (best of 7, whoever wins 4 games first) in which each team has a 50% chance of winning any given round (no draws allowed). What is the probability that the series goes to 7 games?

For the series to go to 7 games, each team must have won exactly three times for the first 6 games, an occurrence having probability

![image](https://user-images.githubusercontent.com/60442877/192422763-d2a942b7-b312-483a-a1ef-b5f728947e0b.png)

where the numerator is the number of ways of splitting up 3 games won by either side, and the denominator is the total number of possible outcomes of 6 games.
