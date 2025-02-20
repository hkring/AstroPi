from exif import Image
from datetime import datetime
from logzero import logger
import math, os
  
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
    
def get_DMScoordinates(image: str) -> str:
    """
    Read Image Meta data and returns degree, minutes, seconds (DMS) coordinates

    Args:
        image: file path to Exif Image

    Returns:
       dms (string): such as 37°25'19.07"N, 122°05'06.24"W     
    """
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lat = img.get('gps_latitude')
        latref = img.get('gps_latitude_ref')
        lon = img.get('gps_longitude')
        lonref = img.get('gps_longitude_ref')
        return "{0}°{1}\'{2}\"{3}, {4}°{5}\'{6}\"{7}".format(int(lat[0]), int(lat[1]), lat[2], latref, int(lon[0]), int(lon[1]), lon[2], lonref)


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
    return c, r*c 

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

#----------------------------------------------------- Main Logic -------------------------------------------------------
def main() -> None:

    R = 6378137   # Radius earth in [m] 
    H = 420000    # ISS orbit height in [m]

    imagerelpath = "./test"

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
    # read filepath from folders
    images.clear()
    # only for testing 
    try:
        os.mkdir(imagerelpath)
    except FileExistsError:
        logger.warning(f"Directory '{imagerelpath}' already exists.")
        if (imagerelpath != "./test") :
            delete_files_in_directory(imagerelpath)
    except PermissionError:
        logger.error(f"Permission denied: Unable to create '{imagerelpath}'.")
        imagerelpath = ""  

    files = [f for f in os.listdir(imagerelpath)] 
    for f in files:
        if imagerelpath == "":
            images.append({"imagepath": f})
        else:
            images.append({"imagepath": imagerelpath + '/' + f})

    # Loop over all images and extract decimal coordinates
    for img in images:
        img.update({"datetime_original":get_time(img.get("imagepath"))})
        img.update({"latitude": get_signedLatCoordinate(img.get("imagepath"))})
        img.update({"longitude": get_signedLonCoordinate(img.get("imagepath"))})

    # Caclulate angular distance. Start the loop with the second image but use the previous image[i-1] to extract the origin
    for i in range(1, len(images), 1):
        pointAlat = images[i-1].get("latitude")
        pointAlon = images[i-1].get("longitude")
        pointBlat = images[i].get("latitude")
        pointBlon = images[i].get("longitude")

        beta, arclength_m = calculate_haversine((R),pointAlat, pointAlon, pointBlat, pointBlon)  
        images[i].update({"arclength_m": arclength_m})
        orbitlength_m = beta * (R + H)
        images[i].update({"orbitlength_m": orbitlength_m})
        deltatime_sec = get_time_difference(images[i-1].get("imagepath"), images[i].get("imagepath"))
        images[i].update({"deltatime_sec": deltatime_sec})
        speed_mpsec = arclength_m / deltatime_sec
        images[i].update({"speed_mpsec": speed_mpsec})

    # Path decimal coordinates
    for img in images:
        print('{:.14f}'.format(img.get("longitude")) + ',' + '{:.14f}'.format(img.get("latitude")) +',0') 

    # Calculate total path length 
    k = 'arclength_m' # key
    segment_m = list(i[k] for i in images if k in i)
    totalpathdistance_m = sum(segment_m)
    logger.debug(f"The path distance is {totalpathdistance_m} in [m]") 

     # Calculate total orbit length 
    k = 'orbitlength_m' # key
    orbitdistance_m = list(i[k] for i in images if k in i)
    totalorbitdistances = sum(orbitdistance_m)
    logger.debug(f"The orbit distance is {totalorbitdistances} in [m]") 

    # Calculate average ground speed
    k = 'speed_mpsec' # key
    pathspeed_mpsec = list(i[k] for i in images if k in i)
    avg_pathspeed_mpsec = sum(pathspeed_mpsec) / len(pathspeed_mpsec)
    logger.debug(f"The average speed is {avg_pathspeed_mpsec/1000} in kmps") 
    
    # Calculate average orbit speed
    k = 'deltatime_sec' # key
    timedifferences_sec = list(i[k] for i in images if k in i)
    totaltimedifferences_sec = sum(timedifferences_sec)
    orbitspeed_mpsec = totalorbitdistances / totaltimedifferences_sec
    logger.debug(f"The calculated orbit speed is {orbitspeed_mpsec/1000} in kmps") 

    # Calculate the period the ISS orbits
    period = 2*math.pi*(R)/(avg_pathspeed_mpsec)
    logger.debug(f"The calculated ISS orbit period is {period/60:.2f} in minutes")

if __name__ == "__main__":
    main()

#----------------------------------------------------- Main Logic-------------------------------------------------------    