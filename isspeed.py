from datetime import datetime
from logzero import logger
from exif import Image
import cv2, math
import pandas as pd

R = 6378137   # Radius earth in [m] 
#GSD = 12648    # ground sample distance in cm/pixel
GSD = 11625

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

def calculate_speed_inkmps(feature_distance: float, gsd: float, time_difference: float):
    '''
    Convert feature distance using ground sample distance to calculate the speed

    Args:
        feature_distance (float): average distance matching key points in pixel
        gsd (float): ground sample distance in cm/pixel
        time_difference (float): difference datetime_original in seconds
    '''
    distance = feature_distance * gsd / 100000
    speed = distance / time_difference
    return speed


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

for img in images:
    img.update({"datetime_original":get_time(img.get("imagepath"))})

# Holger comment: Caclulate angular distance. Start the loop with the second image but use the previous image[i-1] to extract the start point 
for i in range(1, len(images), 1):
    dtime = get_time_difference(images[i-1].get("imagepath"),images[i].get("imagepath"))    
    images[i].update({"dtime": dtime})
    image_1_cv, image_2_cv = convert_to_cv(images[i-1].get("imagepath"),images[i].get("imagepath"))
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
    matches = calculate_matches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
    avg_distance = calculate_mean_distance(coordinates_1, coordinates_2)
    images[i].update({"distance": avg_distance})
    speed = calculate_speed_inkmps(avg_distance, 12648, dtime)
    images[i].update({"speed": speed})

for img in images:
    print(img) 

# Holger comment: calculate total path length 
k = 'distance' # key
seg_distance = list(i[k] for i in images if k in i)

# Holger comment: sum the distance of ALL segments
total = sum(seg_distance)
logger.debug(f"The total feature distance is {total} in pixel") 
logger.debug(f"The total distance is {total*GSD/100000} in km") 

# Holger comment: average ground speed
k = 'speed' # key
seg_speed = list(i[k] for i in images if k in i)
avg_spee = sum(seg_speed) / len(seg_speed)
logger.debug(f"The average speed is {avg_spee} in kmps") 

period = 2*math.pi*R/(1000 * avg_spee)
logger.debug(f"The calculated ISS orbit period is {period/60:.2f} in minutes")
