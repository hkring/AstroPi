from exif import Image
from datetime import datetime
from logzero import logger
import math

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

def get_gps_DMS_String(image: Image) -> str:
    """
    Read Image Meta data and returns location coordinates as Degree, Minutes, Seconds (DMS)

    Args:
        image: file path to Exif Image

    Returns:
        DMS (string): such as 37°25'19.07"N, 122°05'06.24"W     
    """
    with open(image, 'rb') as image_file:
        img = Image(image_file)
        lat = img.get('gps_latitude')
        latref = img.get('gps_latitude_ref')
        lon = img.get('gps_longitude')
        lonref = img.get('gps_longitude_ref')
        return "{0}°{1}\'{2}\"{3}, {4}°{5}\'{6}\"{7}".format(int(lat[0]), int(lat[1]), lat[2], latref, int(lon[0]), int(lon[1]), lon[2], lonref)

# camera properties
widthpx = 4056
focallength = 5 # focal length in [mm]
sensorwidth = 6.17 # sendor width [mm]
beta = math.atan((sensorwidth/2)/focallength) # vertex angle [radian]

def calculateGSDvalue(altidue: int) -> int:
    '''
    Calculates the ground sampling distance (GSD)

    Args: 
        altitude (int): ISS orbits between 370000 - 460000 [meter]

    Return: 
        ground sampling distance (int): scale factor [cm/pixel]   
    '''
    grounddistance = 2 * math.tan(beta) * altidue * 100
    return int(grounddistance/widthpx)

def convert_to_degree(dmscoordinate) -> float:
    return dmscoordinate[0] + (dmscoordinate[1] / 60) + (dmscoordinate[2] / 3600)

def get_sign(refchar: str) -> float:
    match refchar:
        case "N" | "E":
            return 1.0
        case "S" | "W":
            return -1.0
        case _:
            return 1.0 

def degree_to_radian(ref: str, degree: float) -> float:
    return get_sign(ref) * degree * math.pi / 180
     
def measure_distance(image_1, image_2) -> float:
    '''
    Calculate the the great-cicle distance between two point on the basis of spherical earth

    Args:

    '''
    with open(image_1, 'rb') as image_file:
        img = Image(image_file)
        latA = convert_to_degree(img.get('gps_latitude'))
        latrefA = img.get('gps_latitude_ref')
        lonA = convert_to_degree(img.get('gps_longitude'))
        lonrefA = img.get('gps_longitude_ref')
    
    with open(image_2, 'rb') as image_file:
        img = Image(image_file)
        latB = convert_to_degree(img.get('gps_latitude'))
        latrefB = img.get('gps_latitude_ref')
        lonB = convert_to_degree(img.get('gps_longitude'))
        lonrefB = img.get('gps_longitude_ref')   

    R = 6378.137 #Radius earth in km
    dlat = degree_to_radian(latrefB,latB) - degree_to_radian(latrefA,latA)
    dlon = degree_to_radian(lonrefB,lonB) - degree_to_radian(lonrefA,lonA)
    #a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(degree_to_radian(latrefA,latA)) * math.cos(degree_to_radian(latrefB,latB)) * math.sin(dlon/2)*math.sin(dlon/2)
    #c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    a = 0.5 - math.cos(dlat)/2 + math.cos(degree_to_radian(latrefA,latA)) * math.cos(degree_to_radian(latrefB,latB)) * (1-math.cos(dlon))/2
    c = 2* math.asin(math.sqrt(a))
    return R*c
    

#----------------------------------------------------- Main Logic -------------------------------------------------------

#Holger comment: use two photos located in the same folder as your main.py file
image_1 = 'test/photo_0683.jpg'
image_2 = 'test/photo_0684.jpg'

starttime = datetime.now().timestamp()
flag = 0
logger.debug('Start looping')
while datetime.now().timestamp()- starttime < 30:
    # Holger comment : round timestamp to compare with an integer 
    if flag == 0 and round( datetime.now().timestamp() - starttime) == 15:
        logger.debug('15 seconds')
        # Holger comment : extract geo location photo 1
        coordinate = get_gps_DMS_String(image_1)
        logger.debug(f"{image_1} coordinates {coordinate}")        
        flag = 1

# Holger comment : extract geo location photo 2
coordinate = get_gps_DMS_String(image_2)
logger.debug(f"{image_2} coordinates {coordinate}")

# Holger comment : read the timestamp the photos have been taken
time_difference = get_time_difference(image_1,image_2)
logger.debug(f"The time difference between two photos is {time_difference} in seconds")

# Holger comment : calculate distance between geo coordinates
distance = measure_distance(image_1, image_2)
logger.debug(f"The ground distance between two coordinate is {distance} in km")

speed = distance / time_difference
logger.debug(f"Result speed calculation {speed} in kmps")

logger.debug('30 seconds')