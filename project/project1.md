BEGINNING AT A STAKE IN THE SOUTH SIDE OF FLEMING DRIVE 511.4 FEET IN A WESTERLY DIRECTION FROM THE WEST SIDE OF OAK DRIVE AND RUNNING THENCE ALONG AND WITH THE SOUTH SIDE OF FLEMING DRIVE NORTH 76° 52' WEST 130 FEET TO A STAKE; THENCE SOUTH 13° 08' WEST 175 FEET TO A STAKE; THENCE SOUTH 76° 52' EAST 120.2 FEET TO A STAKE; THENCE IN AN EASTERLY DIRECTION WITH THE ARC OF A COUNTERCLOCKWISE CURVE HAVING A RADIUS OF 916.78 FEET, AN ARC DISTANCE OF 9.8 FEET TO A STAKE; THENCE NORTH 13° 08' EAST 175 FEET TO A STAKE, THE POINT AND PLACE OF BEGINNING, AND BEING ALL OF LOT 29 AND A PORTION OF LOTS 28 AND 30, BLOCK A, OF HUCKLEBERRY HEIGHTS EXTENSION #2 SUBDIVISION, AS SHOWN ON THE PLAT RECORDED AT PLAT BOOK 22, PAGE 66, DURHAM COUNTY REGISTRY
import matplotlib.pyplot as plt
import numpy as np
import math

# Convert degrees, minutes, and direction to radians
def dms_to_rad(degrees, minutes, direction):
    decimal_degrees = degrees + minutes / 60.0
    if direction in ["S", "W"]:
        decimal_degrees *= -1
    return math.radians(decimal_degrees)

# Moves in format: (direction, degrees, minutes, distance, is_curve, radius)
moves = [
    ("N", 76, 52, 130, False, None),  # WEST
    ("S", 13, 8, 175, False, None),  # WEST
    ("S", 76, 52, 120.2, False, None),  # EAST
    ("E", None, None, 9.8, True, 916.78),  # curve with given radius
    ("N", 13, 8, 175, False, None)  # EAST
]

# Starting point
x, y = [0], [0]

for move in moves:
    direction, degrees, minutes, distance, is_curve, radius = move
    if is_curve:
        # Handling the curve using arc, assuming it's a quarter of a circle
        angle = 2 * np.pi * distance / (2 * np.pi * radius)
        dx = radius * (1 - np.cos(angle))
        dy = radius * np.sin(angle)
    else:
        angle = dms_to_rad(degrees, minutes, direction)
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)
        
    x.append(x[-1] + dx)
    y.append(y[-1] + dy)

# Plot the property
plt.figure()
plt.plot(x, y, marker='o')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
