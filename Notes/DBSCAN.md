# DBSCAN Summary

* Step 1: Choose an appropriate radius value, and for each data point, in the circle range, count the number of points around it 
* Step 2: Choose an appropriate threshold, and define core points and non-core points
* Step 3: Randomly pick one core point as the first cluster
* Step 4: Within the circle range with the radius value set in step 1, find all the core points that is close to the first cluster, and merge them into first cluster, keep this finding and merging process until we can't find any core point that can be merged into the first cluster anymore
* Step 5: Add non-core points to the first cluster if they are close to it
* Step 6: If we still have core points left, randomly pick a remaining core point as the second cluster, and follow above steps to create the second cluster
* Step 7: Keep creating cluster until no core points left, the remaining non-core points are viewed as outliers

[DBSCAN Detail](https://github.com/yangshiteng/StatQuest-Study-Notes/blob/main/Notes/DBSCAN.pdf)


