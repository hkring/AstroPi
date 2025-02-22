from pathlib import Path
from datetime import datetime
from logzero import logger
import logzero
from astro_pi_orbit import ISS
#from picamzero import Camera
from exif import Image
import cv2, math, os, io
import pandas as pd
import numpy as np

MAX_images = 15
duration        = 600 # main loop durtaion seconds
R   = 6378137   # Radius earth in [m] 

iss = ISS()
#cam = Camera()

base_folder     = Path(__file__).parent.resolve()
starttime       = datetime.now().timestamp()

# Data structure for each image instead of multiple loose variables
# Images -> List (Dictonary)
#   + imagepath (string)            - file path 
#   + datetime_original (datetime)  - image capture time
#   + latitude (floor)              - coordinate in [decimal degree] 
#   + lontitude (floor)             - coordinate in [decimal degree}
#   + geodistance_m (floor)         - arc length between two points in [meter]
#   + featuredistance_pixel (floor) - chord length between two points in [pixel]
#   + deltatime_sec (floor)         - time difference in [seconds]
#   + speed_kmps (floor)            - orbit speed in [meter per seconds]   
images = []

# Set a minimum log level
#logzero.loglevel(logzero.DEBUG)

def get_ISS_coordinates():
    '''
    Retrieve the ISS coordinates.

    returns:
        latlon (tuple(floor)): latitude, lontidtude in degrees, minutes and seconds     
    '''
    point = iss.coordinates()
    return (point.latitude.signed_dms(), point.longitude.signed_dms())

def get_image_width(imagename: str) -> int:
    '''
    Read file meta data 'image_width'.

    args 
        imagename (str): filename
    returns: 
        width (int): image_width
    '''
    with open(imagename, 'rb') as image_file:
        img = Image(image_file)
        width_pixel = img.get("image_width")
        return width_pixel

def get_time(imagename: str) -> datetime: 
    '''
    Read file meta data 'datetime_original'.
    
    args 
        imagename (str): filename
    returns: 
        time (datetime): datetime_original
    ''' 
    with open(imagename, 'rb') as image_file:   
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
    Calculate the distance between the two points along a great circle on a surface of a sphere

    args:
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
    c = 2 * math.asin(math.sqrt(a))
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

def calculate_chordlength(r: float, acrlength: float) -> float:
    '''
    Calculate the distance between two points of a circle 

    args:
        r (float): cicrle radius 
        acrlength (float)
    returns:
        chordlength (float)
    '''
    return 2*r*math.sin(acrlength/(2*r))  

def calculate_chordlength(r: float, acrlength: float) -> float:
    '''
    Calculate the distance between two points of a circle 

    args:
        r (float): cicrle radius 
        acrlength (float)
    returns:
        chordlength (float)
    '''
    return 2*r*math.sin(acrlength/(2*r)) 

def calculate_arclength(r: float, chordlength: float) -> float:
    '''
    Calculate the distance between two points 

    args:
        r (float): cicrle radius 
        chordlength (float)
    returns:
        chordlength (float)
    '''
    return 2*r*math.asin(chordlength/(2*r)) 

def calculate_speed_inkmps(image_width: int, feature_distance: float , time_difference: float, geo_distance: float) -> float:
    '''
    Identify the closest ground sampling rate based on different orbit altitudes to calculate the ISS speed.

    args:
        image_width (int): usually highest resolutiom 4056
        feature_distance (float): length in [pixel]
        time_difference (float): time between two images
        geo_distance (float): length in [m]

    returns:
        speed (float): speed in [kmps]
    '''
    orbitarray_m = [390000, 395000, 400000, 450000, 410000, 415000, 420000, 425000, 430000, 435000]
    array_m = []
    for o in orbitarray_m:
        arclength_m = calculate_arclength(R, feature_distance*calculate_ground_sampling_distance(image_width, o)/100)
        array_m.append(arclength_m)

    mapbyorbit = np.vstack((np.array(orbitarray_m),np.array(array_m)))  
    idx, closestarclength_m = find_closest_value(mapbyorbit[1,:], geo_distance)
    logger.debug(f'Function calculate_speed_inkmps - Found orbit {orbitarray_m[idx]} closest distance {closestarclength_m/1000} in km"') 
    arclength_m = closestarclength_m * (R + orbitarray_m[idx])/R 
    
    return arclength_m/(1000*time_difference)


def next_image(i: int) -> None:
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
#    imagename = imagerelpath + f'gps_image{i:02d}.jpg'   
#    cam.take_photo(imagename, get_ISS_coordinates())
# !!!!!!!!!!!!!!!To remove only allowed during testing!!!!!!!!!!!!!!!!!!!
    files = [f for f in os.listdir(base_folder) if f.endswith(".jpg")] 
    imagename = files[i]
# !!!!!!!!!!!!!!!To remove only allowed during testing!!!!!!!!!!!!!!!!!!!
    logger.info("Take a new photo " + imagename)    
    images.append({"imagepath": imagename})

def image_update(previousimage, thisimage) -> None:
    logger.debug(f'Function image_update - process section between')
    logger.debug(f'Function image_update - previous image  {previousimage} ')
    
    originAlat = previousimage.get("latitude")
    originAlon = previousimage.get("longitude")
    pointBlat = thisimage.get("latitude")
    pointBlon = thisimage.get("longitude")
    geodistance_m = calculate_haversine((R), originAlat, originAlon, pointBlat, pointBlon)  
    thisimage.update({"geodistance_m": geodistance_m})
    deltatime_sec = get_time_difference(previousimage.get("imagepath"),thisimage.get("imagepath"))    
    thisimage.update({"deltatime_sec": deltatime_sec})
    image_1_cv, image_2_cv = convert_to_cv(previousimage.get("imagepath"),thisimage.get("imagepath"))
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    featuredistance_pixel = calculate_mean_distance(coordinates_1, coordinates_2)
    thisimage.update({"featuredistance_pixel": featuredistance_pixel})
    speed_kmps = calculate_speed_inkmps(get_image_width(thisimage.get("imagepath")),featuredistance_pixel,deltatime_sec, geodistance_m) 
    thisimage.update({"speed_kmps": speed_kmps})
    logger.debug(f'Function image_update - and next image  {thisimage} ')
    logger.debug(f'Function image_update finished!')

def find_closest_value(values, K):    
     values = np.asarray(values)
     idx = (np.abs(values - K)).argmin()
     return idx, values[idx]   
#----------------------------------------------------- Main Logic -------------------------------------------------------
def main() -> None:

    logger.debug('Function main started ....')
    # read filepath from folders
    images.clear()

    lastPictureTime = 0
    while datetime.now().timestamp() - starttime < duration:
        # no more imgages allowed
        if(len(images) >= MAX_images):
            break
        
        # get new picture every 15 seconds
        if lastPictureTime == 0 or datetime.now().timestamp() - lastPictureTime >= 2:
            next_image(len(images))
            lastPictureTime = datetime.now().timestamp() 
            thisimage = images[len(images)-1]
            thisimage.update({"datetime_original":get_time(thisimage.get("imagepath"))})
            thisimage.update({"latitude": get_signedLatCoordinate(thisimage.get("imagepath"))})
            thisimage.update({"longitude": get_signedLonCoordinate(thisimage.get("imagepath"))})
            
            if (len(images) > 1):
                previousimage = images[len(images)-2]
                image_update(previousimage, thisimage)
          
    # Calculate total feature distance
    k = 'featuredistance_pixel' # key
    featuredistance_pixel = list(i[k] for i in images if k in i)

    # Sum the distance of ALL segments
    totaldistance_pixels = sum(featuredistance_pixel)
    logger.info(f"The total feature distance is {totaldistance_pixels} in pixel") 
   
    k = 'geodistance_m' # key
    angulardistance_m = list(i[k] for i in images if k in i)
    totalpathdistance_m = sum(angulardistance_m)
    logger.info(f"The geo path distance is {totalpathdistance_m/1000} in km") 

    k = 'speed_kmps' # key
    speed_kmps = angulardistance_m = list(i[k] for i in images if k in i)
    avg_speed_kmps = calculate_mean(speed_kmps)

    resultfilepath = "./result.txt"
    resultspeed_kmps = "{:.5f}".format(avg_speed_kmps)
    with io.open(resultfilepath, 'w') as file:
        file.write(resultspeed_kmps)

    logger.info(f"Function main - Result speed {resultspeed_kmps} written to {resultfilepath}")

# todo calculate the orbit period based on the estimate orbit altitude
 #   period = 0.0
 #   if(avg_speed_kmps > 0):
 #       period = 2*math.pi*(R + orbitlist_m[idx])/(1000 * avg_speed_kmps)
 #   logger.debug(f"The calculated ISS orbit period is {period/60:.2f} in minutes")

    # Path decimal coordinates
    geolocation = []
    for img in images:
        geolocation.append(str('{:.14f}'.format(img.get("longitude"))) + "," + str('{:.14f}'.format(img.get("latitude")) + ",0"+ "\n"))

    geolocfilepath = "./geoloc.txt"
    with io.open(geolocfilepath, 'w') as file:
        file.writelines(geolocation)

# entree point to execute main logic 
if __name__ == "__main__":
    main()

 #----------------------------------------------------- Main Logic -------------------------------------------------------    