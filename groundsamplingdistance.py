import math

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

#----------------------------------------------------- Main Logic -------------------------------------------------------
def main() -> None:
    issorbbitheight = 420000    # ISS orbit height [m]
    imagewidth      = 4056      # photo width in [pixels]
    R               = 6378137   # Radius earth in [m]    
    gsd_cmppixel = calculate_ground_sampling_distance(imagewidth, issorbbitheight)
    print(f'Ground sampling distance {gsd_cmppixel} in [cm per pixel] for ISS orbit {issorbbitheight} in [m]')

    # validating with an arc length = 61.044345379960895 km
    distance_km = calculate_distance(R, 61.044345379960895) # distance between two points in [km]
    print(f'Calculated feature distance is {100000 * distance_km/gsd_cmppixel} in [pixel]')

if __name__ == "__main__":
    main()

 #----------------------------------------------------- Main Logic -------------------------------------------------------      