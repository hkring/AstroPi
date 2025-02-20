import math
import numpy as np

fl              = 5         # focal length in [mm]
sw              = 6.287     # sendor width [mm]

def calculate_distance(r_m: float, acrlen_m: float) -> float:
    '''
    Calculate the distance between two point on an arc

    Args:
        r_m (float): arc radius in [meter]
        arclen_m (float): arc segment length in [meter]
    Returns:
        d_m (float): distance in [meter      
    '''
    return 2*r_m*math.sin(acrlen_m/(2*r_m))

def calculate_ground_sampling_distance(imagewidth_pixels: int, orbitheight_m:float) -> float: 
    '''
    Calculate the Image width footprint on the ground

    Args: 
        imagewidth_pixels (int): Image width in [pixels]
        orbitheight_m (float):  ISS orbit height in [meter]    
    Returns
        gsd_cmppixel (float): ground sampling distances in [cm/pixel]     
    '''
    dw = 2* ((sw/2)/fl) * orbitheight_m #Image width footprint on the ground in [m]
    return dw*100/imagewidth_pixels 

def find_closest_value(values, K):    
     values = np.asarray(values)
     idx = (np.abs(values - K)).argmin()
     return idx, values[idx]

#----------------------------------------------------- Main Logic -------------------------------------------------------
def main() -> None:
#    issorbbitheight = 420000    # ISS orbit height [m]
    imagewidth      = 4056      # photo width in [pixels]
    R               = 6378137   # Radius earth in [m]  
    
    angulardistance_m = 61044.3453799609
    featuredistance_pixels = 489.7401280888609

    orbitlist_m = [390000, 400000, 410000, 420000, 430000]
    calcdistances_pixels = []
    for o in orbitlist_m:
        calcdistances_pixels.append(calculate_distance(R, angulardistance_m) * 100/ calculate_ground_sampling_distance(imagewidth, o))
        print(calcdistances_pixels)

    list = np.vstack((np.array(orbitlist_m),np.array(calcdistances_pixels)))
    idx, distance_pixel = find_closest_value(list[1,:], featuredistance_pixels)

    print(f'Calculated feature distance is {distance_pixel} in [pixel] by orbit {orbitlist_m[idx]} in [m]')

if __name__ == "__main__":
    main()

 #----------------------------------------------------- Main Logic -------------------------------------------------------      