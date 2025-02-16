from exif import Image
from datetime import datetime
from logzero import logger
import math, os

R = 6378137   # Radius earth in [m] 
duration = 30 # seconds
starttime = datetime.now().timestamp()

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

def calculate_haversine(originAlat: float, originAlon: float, pointBlat: float, pointBlon: float) -> float:
    """
    Calculate the arc length between two points on a surface of a sphere

    Args:
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
    return R*c 

# Holger comment: Defining a data structure for each image instead of multiple loose variables
# Images -> List (Dictonary)
#   + imagepath (string)    - file path 
#   + latlon (tuple[floor]) - decimal coordinate
#   + distance (floor)      - angular distance between two points   
images = [
    {"imagepath":  "test/photo_0673.jpg"},
    {"imagepath":  "test/photo_0674.jpg"}#,
    #{"imagepath":  "test/photo_0675.jpg"},
    #{"imagepath":  "test/photo_0676.jpg"},
    #{"imagepath":  "test/photo_0678.jpg"},
    #{"imagepath":  "test/photo_0679.jpg"},
    #{"imagepath":  "test/photo_0680.jpg"},
    #{"imagepath":  "test/photo_0681.jpg"},
    #{"imagepath":  "test/photo_0682.jpg"},
    #{"imagepath":  "test/photo_0683.jpg"},
    #{"imagepath":  "test/photo_0684.jpg"},
    #{"imagepath":  "test/photo_0685.jpg"},
    #{"imagepath":  "test/photo_0687.jpg"}
    ]

#----------------------------------------------------- Main Logic -------------------------------------------------------

# Holger comment: read filepath from folders
images.clear()
imagerelpath = "./test"
files = [f for f in os.listdir(imagerelpath)] 
for f in files:
    if imagerelpath == "":
        images.append({"imagepath": f})
    else:
        images.append({"imagepath": imagerelpath + '/' + f})

# Holger comment: Loop over all images and extract decimal coordinates
for img in images:
    img.update({"datetime_original":get_time(img.get("imagepath"))})
    img.update({"latitude": get_signedLatCoordinate(img.get("imagepath"))})
    img.update({"longitude": get_signedLonCoordinate(img.get("imagepath"))})

# Holger comment: Caclulate angular distance. Start the loop with the second image but use the previous image[i-1] to extract the start point 
for i in range(1, len(images), 1):
    pointAlat = images[i-1].get("latitude")
    pointAlon = images[i-1].get("longitude")
    pointBlat = images[i].get("latitude")
    pointBlon = images[i].get("longitude")

    arclength = calculate_haversine(pointAlat, pointAlon, pointBlat, pointBlon)  
    images[i].update({"arclength": arclength})
    dtime = get_time_difference(images[i-1].get("imagepath"), images[i].get("imagepath"))
    images[i].update({"dtime": dtime})
    speed = arclength / dtime
    images[i].update({"speed": speed})

# Holger comment: path decimal coordinates
for img in images:
    print('{:.14f}'.format(img.get("latitude")), '{:.14f}'.format(img.get("longitude"))) 

for img in images:
    print(img) 

# Holger comment: calculate total path length 
k = 'arclength' # key
seg_d = list(i[k] for i in images if k in i)

# Holger comment: sum the distance of ALL segments
total = sum(seg_d)
logger.debug(f"The path distance is {total} in [m]") 

# Holger comment: average ground speed
k = 'speed' # key
seg_speed = list(i[k] for i in images if k in i)
avg_spee = sum(seg_speed) / (1000 * len(seg_speed))
logger.debug(f"The average speed is {avg_spee} in kmps") 

period = 2*math.pi*R/(avg_spee)
logger.debug(f"The calculated ISS orbit period is {period/60:.2f} in minutes")
