from exif import Image
from datetime import datetime
from logzero import logger
import math

R = 6378.137 #Radius earth in km

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

def get_signedLatLonCoordinate(image: str) -> tuple:
    """
    Read Image Meta data and returns signed decimal coordinates

    Args:
        image: file path to Exif Image

    Returns:
        decimal (tuble[floor]): {lat, lon}
    """
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lat = img.get('gps_latitude')
        latref = img.get('gps_latitude_ref')
        lon = img.get('gps_longitude')
        lonref = img.get('gps_longitude_ref')
        signedlat = get_sign(latref) * convert_to_degree(lat)
        signedlon = get_sign(lonref) * convert_to_degree(lon)
        return tuple({signedlat, signedlon})

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

def calculate_haversine(pointA: tuple[float], pointB: tuple[float]) -> float:
    """
    Calculate the angular distance between two points on a surface of a sphere

    Args:
        pointA (tuble(float)): start point
        pointB (tuble(float)): end point

    return distance (float)     
    """ 
    dlat = convert_degreeToRadian(pointB[0]) - convert_degreeToRadian(pointA[0])
    dlon = convert_degreeToRadian(pointB[1]) - convert_degreeToRadian(pointA[1])
    a = 0.5 - math.cos(dlat)/2 + math.cos(convert_degreeToRadian(pointA[0])) * math.cos(convert_degreeToRadian(pointB[0])) * (1-math.cos(dlon))/2
    c = 2* math.asin(math.sqrt(a))
    return R*c 
#----------------------------------------------------- Main Logic -------------------------------------------------------

# Holger comment: Defining a data structure for each image instead of multiple loose variables
# Images -> List (Dictonary)
#   + imagepath (string)    - file path 
#   + latlon (tuple[floor]) - decimal coordinate
#   + distance (floor)      - angular distance between two points   
images = [
    {"imagepath":  "test/photo_0673.jpg"},
    {"imagepath":  "test/photo_0674.jpg"},
    {"imagepath":  "test/photo_0675.jpg"},
    {"imagepath":  "test/photo_0676.jpg"},
    {"imagepath":  "test/photo_0678.jpg"},
    {"imagepath":  "test/photo_0679.jpg"},
    {"imagepath":  "test/photo_0680.jpg"},
    {"imagepath":  "test/photo_0681.jpg"},
    {"imagepath":  "test/photo_0682.jpg"},
    {"imagepath":  "test/photo_0683.jpg"},
    {"imagepath":  "test/photo_0684.jpg"},
    {"imagepath":  "test/photo_0685.jpg"},
    {"imagepath":  "test/photo_0687.jpg"}
    ]

# Holger comment: Loop over all images and extract decimal coordinates
for img in images:
    img.update({"datetime_original":get_time(img.get("imagepath"))})
    img.update({"latlon": get_signedLatLonCoordinate(img.get("imagepath"))})

# Holger comment: Caclulate angular distance. Start the loop with the second image but use the previous image[i-1] to extract the start point 
for i in range(1, len(images), 1):
    distance = calculate_haversine(images[i-1].get("latlon"), images[i].get("latlon"))  
    images[i].update({"distance": distance})
    timedifference = get_time_difference(images[i-1].get("imagepath"), images[i].get("imagepath"))
    images[i].update({"timedifference": timedifference})
    speed = distance / timedifference
    images[i].update({"speed": speed})

# Holger comment: Loop over agian all images
for img in images:
    distance = img.get("distance")
    logger.debug(f"The ground distance between two coordinate is {distance} in km")
    timedifference = img.get("timedifference")
    logger.debug(f"The time difference between two photos is {timedifference} in seconds")
    speed = img.get("speed")
    logger.debug(f"The calculated speed is {speed} in kmps")
    #print(img) 
