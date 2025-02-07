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

    Args altitude (int): ISS orbits between 370000 - 460000 [meter]

    Return GSD (int): scale factor [cm/pixel]   
    '''
    grounddistance = 2 * math.tan(beta) * altidue * 100
    return int(grounddistance/widthpx)

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
        coordinate = get_gps_DMS_String(image_1)
        logger.debug(f"{image_1} coordinates {coordinate}")        
        flag = 1

coordinate = get_gps_DMS_String(image_2)
logger.debug(f"{image_2} coordinates {coordinate}")
#Holger comment: open image files and read timestamp meta data  
time_difference = get_time_difference(image_1,image_2)
logger.debug(f"The time difference between two photos is {time_difference} seconds")
#Holger comment: assuming an altitude of 435250  
gds = calculateGSDvalue(435250)
logger.debug(f"GSD Scaling factor is {gds}")

logger.debug('30 seconds')