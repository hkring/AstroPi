from datetime import datetime
from logzero import logger
import logzero
from astro_pi_orbit import ISS
from picamzero import Camera
from exif import Image
import cv2, math, os, io
import pandas as pd
import numpy as np

MAX_images = 42
iss = ISS()
cam = Camera()
duration        = 600 # main loop durtaion seconds
starttime       = datetime.now().timestamp()

imagerelpath    = "./" 

R   = 6378137   # Radius earth in [m] 
#H   = 390000    # ISS orbit height in [m]

# Data structure for each image instead of multiple loose variables
# Images -> List (Dictonary)
#   + imagepath (string)           - file path 
#   + datetime_original (datetime) - image capture time
#   + latitude (floor)             - coordinate in [decimal degree] 
#   + lontitude (floor)            - coordinate in [decimal degree}
#   + arclength_m (floor)          - arc length between two points in [meter]
#   + deltatime_sec (floor)        - time difference in [seconds]
#   + ground_speed_mpsec (floor)   - ground speed in [meter per seconds]   
images = []

# Set a minimum log level
logzero.loglevel(logzero.INFO)

def get_ISS_coordinates():
    '''
    Retrieve the ISS coordinates degree, minutes and seconds
    Args:
        iss (object): astro_pi_orbit object represents ISS
    Returns:
        latlon (tuple): latitude, lontidtude in degree, minutes and seconds     
    '''
    point = iss.coordinates()
    return (point.latitude.signed_dms(), point.longitude.signed_dms())

def get_image_width(image):
     with open(image, 'rb') as image_file:
        img = Image(image_file)
        return img.get("image_width")

def get_time(image):
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        time_str = img.get("datetime_original")
        time = datetime.strptime(time_str, '%Y:%m:%d %H:%M:%S')
        return time

def get_time_difference(image_1, image_2):
    time_1 = get_time(image_1)
    time_2 = get_time(image_2)
    time_difference = time_2 - time_1
    return time_difference.seconds

def get_sign(refchar: str) -> float:
    '''
    Convert a direction character degree minutes seconds (DMS) coordinates 

    Args:
        drection (str): N, E, S or W

    Returns: 
        signed (float): -1.0 for N or E, 1.0 for S or W       
    '''
    match refchar:
        case "N" | "E":
            return 1.0
        case "S" | "W":
            return -1.0
        case _:
            return 1.0 

def convert_to_degree(dms: tuple[float]) -> float:
    '''
    Convert degree minutes seconds (DMS) to decimal coordinates

    Args:
        dms (tuple[float]): degree, minute, seconds

    Returns:
        decimal (float): coordinate not signed     
    '''
    return dms[0] + (dms[1] / 60) + (dms[2] / 3600)

def get_signedLatCoordinate(image: str) -> float:
    """
    Read Image Meta data and returns signed decimal coordinates

    Args:
        image (string): file path to Exif Image

    Returns:
        decimal (floor): latitude
    """
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lat = img.get('gps_latitude')
        latref = img.get('gps_latitude_ref')
        signedlat = get_sign(latref) * convert_to_degree(lat)
        return signedlat

def get_signedLonCoordinate(image: str) -> float:
    """
    Read Image Meta data and returns signed decimal coordinates

    Args:
        image (string): file path to Exif Image

    Returns:
        decimal (floor): longitude 
    """
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lon = img.get('gps_longitude')
        lonref = img.get('gps_longitude_ref')
        signedlon = get_sign(lonref) * convert_to_degree(lon)
        return signedlon

def convert_degreeToRadian(degree: float) -> float:
    '''
    Convert an angle from degree to radiant units

    Args:
        degree (float): 0 - 360  

    Returns:
        radian (float): 0 - 2*pi
    '''
    return degree * math.pi / 180

def calculate_haversine(r: float, originAlat: float, originAlon: float, pointBlat: float, pointBlon: float) -> float:
    """
    Calculate the arc length between two points on a surface of a sphere

    Args:
        r (float): radius of the sphere 
        originAlat (float): origin latitide in [decimal degree]
        originAlon (float): origin longitude in [decimal degree]
        pointBlat (float): point latitide in [decimal degree]
        pointBlon (float): point longitude in [decimal degree]

    return arc length (float)     
    """ 
    dlat = convert_degreeToRadian(pointBlat) - convert_degreeToRadian(originAlat)
    dlon = convert_degreeToRadian(pointBlon) - convert_degreeToRadian(originAlon)
    a = 0.5 - math.cos(dlat)/2 + math.cos(convert_degreeToRadian(originAlat)) * math.cos(convert_degreeToRadian(pointBlat)) * (1-math.cos(dlon))/2
    c = 2* math.asin(math.sqrt(a))
    return r*c 
    
def convert_to_cv(image_1, image_2):
    image_1_cv = cv2.imread(image_1, 0)
    image_2_cv = cv2.imread(image_2, 0)
    return image_1_cv, image_2_cv

def calculate_features(image_1_cv, image_2_cv, feature_number):
    orb = cv2.ORB_create(nfeatures = feature_number)
    keypoints_1, descriptors_1 = orb.detectAndCompute(image_1_cv, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(image_2_cv, None)
    return keypoints_1, keypoints_2, descriptors_1, descriptors_2

def calculate_matches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    return matches

def display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches):
    match_img = cv2.drawMatches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches[:100], None)
    resize = cv2.resize(match_img, (1600,600), interpolation = cv2.INTER_AREA)
    cv2.imshow('matches', resize)
    cv2.waitKey(0)
    cv2.destroyWindow('matches')

def find_matching_coordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1, y1) = keypoints_1[image_1_idx].pt
        (x2, y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1)) 
        coordinates_2.append((x2,y2))  
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    '''
    Calculate the Euclidean distance from origin to a point in [pixel].
    Remove outliers based on quantiles.   
    '''
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    distances = []
    for coordinates in merged_coordinates:
        x_difference = coordinates[0][0] - coordinates[1][0]
        y_difference = coordinates[0][1] - coordinates[1][1]
        distances.append(math.hypot(x_difference, y_difference))
    return calculate_mean(distances)

def calculate_mean(values) -> float:
    d = pd.Series(values)
    outliers = d.between(d.quantile(.10), d.quantile(.90))
    return d[outliers].mean()

def calculate_ground_sampling_distance(imagewidth_pixels: int, orbitheight_m:float) -> float: 
    '''
    Calculate the Image width footprint on the ground. 
    Assuming that the focal lenght (fl) is 5mm and sensor width (sw) = 6.287mm are constant. 

    Args: 
        imagewidth_pixels (int): Image width in [pixels]
        orbitheight_m (float):  ISS orbit height in [meter]    
    Returns
        gsd_cmppixel (float): ground sampling distances in [cm/pixel]     
    '''
    fl              = 5         # focal length in [mm]
    sw              = 6.287     # sendor width [mm]
    dw = 2* ((sw/2)/fl) * orbitheight_m #Image width footprint on the ground in [m]
    return dw*100/imagewidth_pixels 

def calculate_speed_inkmps(image_width, feature_distance: float , time_difference: float, K: float):
    '''
    Calculates speed based on feature pixel distance. Find the approximate orbit by comparing with known value. 
    '''
    orbitarray_m = [390000, 400000, 410000, 420000, 430000]
    distancearray_m = []
    for o in orbitarray_m:
        distancearray_m.append(feature_distance*calculate_ground_sampling_distance(image_width, o)/100)

    mapdistancebyorbit = np.vstack((np.array(orbitarray_m),np.array(distancearray_m)))  
    idx, closestdistance_m = find_closest_value(mapdistancebyorbit[1,:], K)
    logger.debug(f'Found orbit {orbitarray_m[idx]} closest distance {closestdistance_m/1000} in km"') 
    # todo transform to arc later !!
    orbitdistance_m = closestdistance_m * (R + orbitarray_m[idx])/R 
    avg_speed_kmps = orbitdistance_m/(1000*time_difference)
    
    return avg_speed_kmps

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

def next_image(i: int) -> None:
    imagename = imagerelpath + f'gps_image{i:02d}.jpg'   
    cam.take_photo(imagename, get_ISS_coordinates())
    logger.info("Take a new photo " + imagename)    
    images.append({"imagepath": imagename})

def image_update(previousimage, thisimage) -> None:
    logger.debug(f'Function image_update - process section between {previousimage} and {thisimage}')
    originAlat = previousimage.get("latitude")
    originAlon = previousimage.get("longitude")
    pointBlat = thisimage.get("latitude")
    pointBlon = thisimage.get("longitude")
    angulardistance_m = calculate_haversine((R), originAlat, originAlon, pointBlat, pointBlon)  
    thisimage.update({"angulardistance_m": angulardistance_m})
    deltatime_sec = get_time_difference(previousimage.get("imagepath"),thisimage.get("imagepath"))    
    thisimage.update({"deltatime_sec": deltatime_sec})
    image_1_cv, image_2_cv = convert_to_cv(previousimage.get("imagepath"),thisimage.get("imagepath"))
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    featuredistance_pixel = calculate_mean_distance(coordinates_1, coordinates_2)
    thisimage.update({"featuredistance_pixel": featuredistance_pixel})
    speed_kmps = calculate_speed_inkmps(get_image_width(thisimage.get("imagepath")),featuredistance_pixel,deltatime_sec, angulardistance_m) 
    thisimage.update({"speed_kmps": speed_kmps})
    logger.debug(f'Function image_update finished!')

def find_closest_value(values, K):    
     values = np.asarray(values)
     idx = (np.abs(values - K)).argmin()
     return idx, values[idx]   
#----------------------------------------------------- Main Logic -------------------------------------------------------
def main() -> None:

    logger.debug('Begin Main Logic ....')
    # read filepath from folders
    images.clear()

    lastPictureTime = 0
    while datetime.now().timestamp() - starttime < duration:
        # no more imgages allowed
        if(len(images) >= MAX_images):
            break
        
        # get new picture every 15 seconds
        if lastPictureTime == 0 or datetime.now().timestamp() - lastPictureTime >= 10:
            next_image(len(images))
            lastPictureTime = datetime.now().timestamp() 
            thisimage = images[len(images)-1]
            thisimage.update({"datetime_original":get_time(thisimage.get("imagepath"))})
            thisimage.update({"latitude": get_signedLatCoordinate(thisimage.get("imagepath"))})
            thisimage.update({"longitude": get_signedLonCoordinate(thisimage.get("imagepath"))})
            if (len(images) > 1):
                previousimage = images[len(images)-2]
                image_update(previousimage, thisimage)
          

    k = 'speed_kmps' # key
    listspeed_kmps = list(i[k] for i in images if k in i)
    avgspeed_kmps = calculate_mean(listspeed_kmps)

    resultfilepath = "./result.txt"
    resultspeed_kmps = "{:.5f}".format(avgspeed_kmps)
    with io.open(resultfilepath, 'w') as file:
        file.write(resultspeed_kmps)

    logger.info(f"Result speed {resultspeed_kmps} written to {resultfilepath}")

# entree point to execute main logic 
if __name__ == "__main__":
    main()

 #----------------------------------------------------- Main Logic -------------------------------------------------------    