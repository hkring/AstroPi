import math

R = 6378137             # Radius earth in [m] 
altitude    = 420000    # ISS orbit height [m]    
imagewidth  = 4056      # photo width in [pixels]
focallength = 5         # focal length in [mm]
sensorwidth = 6.287     # sendor width [mm]

def calculate_distance(r: float, acrlen: float) -> float:
    return 2*r*math.sin(acrlen/(2*r))

dw = 2* ((sensorwidth/2)/focallength) * altitude #Image width footprint on the ground in [m]
print(dw) 
GSD = dw *100 /imagewidth # Ground sample distance in [cm/pixel]
print(GSD) 

distance = calculate_distance(R, 61.044345379960895) # distance between two points in [km]

print(100000 * distance/GSD)