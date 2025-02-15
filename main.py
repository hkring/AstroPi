from datetime import datetime
from logzero import logger
from astro_pi_orbit import ISS
from picamzero import Camera
from exif import Image
import cv2, math, os, io
import pandas as pd

iss = ISS()
cam = Camera()
duration = 30 # seconds
starttime = datetime.now().timestamp()
imagerelpath = "./photos"#"./photos" #./test

R   = 6378.137 #Radius earth in km
#GSD = 12648    # (4056, 3040) ground sample distance in cm/pixel
GSD = 25297    # (2028, 1520) ground sample distance in cm/pixel

def get_gps_coordinates(iss):
    point = iss.coordinates()
    return (point.latitude.signed_dms(), point.longitude.signed_dms())

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

def calculate_haversine(pointAlat: float, pointAlon: float, pointBlat: float, pintBlon: float) -> float:
    """
    Calculate the angular distance between two points on a surface of a sphere

    Args:
        pointA (tuble(float)): start point
        pointB (tuble(float)): end point

    return distance (float)     
    """ 
    dlat = convert_degreeToRadian(pointBlat) - convert_degreeToRadian(pointAlat)
    dlon = convert_degreeToRadian(pintBlon) - convert_degreeToRadian(pointAlon)
    a = 0.5 - math.cos(dlat)/2 + math.cos(convert_degreeToRadian(pointAlat)) * math.cos(convert_degreeToRadian(pointBlat)) * (1-math.cos(dlon))/2
    c = 2* math.asin(math.sqrt(a))
    return R*c 
    
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
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    distances = []
    for coordinates in merged_coordinates:
        x_difference = coordinates[0][0] - coordinates[1][0]
        y_difference = coordinates[0][1] - coordinates[1][1]
        distances.append(math.hypot(x_difference, y_difference))
    d = pd.Series(distances)
    outliers = d.between(d.quantile(.05), d.quantile(.95))
    #print(str(d[outliers].size) + "/" + str(d.size) + " data points remain.")
    return d[outliers].mean() 

def calculate_speed_inkmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

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

# Holger comment: Defining a data structure for each image instead of multiple loose variables
# Images -> List (Dictonary)
#   + imagepath (string)    - file path 
#   + latlon (tuple[floor]) - decimal coordinate
#   + distance (floor)      - angular distance between two points   
images = [{"imagepath":  "test/photo_0673.jpg"}] #example
images.clear()

try:
    os.mkdir(imagerelpath)
except FileExistsError:
    logger.warning(f"Directory '{imagerelpath}' already exists.")
    if (imagerelpath != "./test") :
        delete_files_in_directory(imagerelpath)
except PermissionError:
    logger.error(f"Permission denied: Unable to create '{imagerelpath}'.")
    imagerelpath = ""  

i = 1
lastPictureTime = 0
while datetime.now().timestamp() - starttime < duration:
    
    # get new picture every 15 seconds
    if lastPictureTime == 0 or datetime.now().timestamp() - lastPictureTime >= 15:
        imagename = imagerelpath + f'/gps_image{i:02d}.jpg'
        logger.debug("Take a new shot " + imagename)
        cam.take_photo(imagename, gps_coordinates=get_gps_coordinates(iss))
        i += 1
        lastPictureTime = datetime.now().timestamp()    

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
    angulardistance = calculate_haversine(pointAlat, pointAlon, pointBlat, pointBlon)  
    images[i].update({"angulardistance": angulardistance})
    dtime = get_time_difference(images[i-1].get("imagepath"),images[i].get("imagepath"))    
    images[i].update({"dtime": dtime})
    image_1_cv, image_2_cv = convert_to_cv(images[i-1].get("imagepath"),images[i].get("imagepath"))
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    pixeldistance = calculate_mean_distance(coordinates_1, coordinates_2)
    images[i].update({"pixeldistance": pixeldistance})
    speed = calculate_speed_inkmps(pixeldistance, GSD, dtime)
    images[i].update({"speed": speed})

# Holger comment: calculate total path length 
k = 'pixeldistance' # key
pixels = list(i[k] for i in images if k in i)

# Holger comment: sum the distance of ALL segments
totalpixels = sum(pixels)
logger.debug(f"The total feature distance is {totalpixels} in pixel") 
logger.debug(f"The calculated distance is {totalpixels*GSD/100000} in km") 

k = 'angulardistance' # key
geodistance = list(i[k] for i in images if k in i)
totalgeo = sum(geodistance)
logger.debug(f"The geo path distance is {totalgeo} in km") 

# Holger comment: average ground speed
k = 'speed' # key
seg_speed = list(i[k] for i in images if k in i)

avg_speed = 0.0
if len(seg_speed) > 0:
    avg_speed = sum(seg_speed) / len(seg_speed)
logger.debug(f"The average speed is {avg_speed} in kmps") 

period = 0.0
if(avg_speed > 0):
    period = 2*math.pi*R/(avg_speed)
logger.debug(f"The calculated ISS orbit period is {period/60:.2f} in minutes")

resultfilepath = "./result.txt"
resultspeed = "{:.4f}".format(avg_speed)
with io.open(resultfilepath, 'w') as file:
    file.write(resultspeed)

logger.debug(f"Result speed {resultspeed} written to {resultfilepath}")

# Holger comment: path decimal coordinates
geolocation = []
for img in images:
    geolocation.append(str('{:.14f}'.format(img.get("longitude"))) + "," + str('{:.14f}'.format(img.get("latitude")) + ",0"+ "\n"))

geolocfilepath = "./geoloc.txt"
with io.open(geolocfilepath, 'w') as file:
    file.writelines(geolocation)
 