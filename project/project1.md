BEGINNING AT A STAKE IN THE SOUTH SIDE OF FLEMING DRIVE 511.4 FEET IN A WESTERLY DIRECTION FROM THE WEST SIDE OF OAK DRIVE AND RUNNING THENCE ALONG AND WITH THE SOUTH SIDE OF FLEMING DRIVE NORTH 76° 52' WEST 130 FEET TO A STAKE; THENCE SOUTH 13° 08' WEST 175 FEET TO A STAKE; THENCE SOUTH 76° 52' EAST 120.2 FEET TO A STAKE; THENCE IN AN EASTERLY DIRECTION WITH THE ARC OF A COUNTERCLOCKWISE CURVE HAVING A RADIUS OF 916.78 FEET, AN ARC DISTANCE OF 9.8 FEET TO A STAKE; THENCE NORTH 13° 08' EAST 175 FEET TO A STAKE, THE POINT AND PLACE OF BEGINNING, AND BEING ALL OF LOT 29 AND A PORTION OF LOTS 28 AND 30, BLOCK A, OF HUCKLEBERRY HEIGHTS EXTENSION #2 SUBDIVISION, AS SHOWN ON THE PLAT RECORDED AT PLAT BOOK 22, PAGE 66, DURHAM COUNTY REGISTRY
import math

    # Convert degrees and minutes to decimal degrees
    degrees = 13
    minutes = 8
    decimal_degrees = degrees + minutes / 60
    
    # Convert decimal degrees to radians
    angle = math.radians(decimal_degrees)
    
    # Compute the change in x and y
    distance = 175
    dx = distance * math.sin(angle)  # WEST decreases x value
    dy = distance * math.cos(angle)  # SOUTH decreases y value
    
    # Starting point
    start_x = 2
    start_y = 3
    
    # Compute the end point
    end_x = start_x - dx
    end_y = start_y - dy
    
    print(f"End point: ({end_x}, {end_y})")

