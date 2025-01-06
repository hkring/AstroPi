from datetime import datetime
from logzero import logger
from astro_pi_orbit import ISS
from picamzero import Camera
from exif import Image
import cv2
import math

iss = ISS()
cam = Camera()
duration = 30 # seconds
starttime = datetime.now().timestamp()

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
        (x2, y2) = keypoints_1[image_2_idx].pt
        coordinates_1.append((x1,y1)) 
        coordinates_2.append((x2,y2))  
    return coordinates_1, coordinates_2

def calculate_mean_distance(coordinates_1, coordinates_2):
    all_distance = 0
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for coordinates in merged_coordinates:
        x_difference = coordinates[0][0] - coordinates[1][0]
        y_difference = coordinates[0][1] - coordinates[1][1]
        distance = math.hypot(x_difference, y_difference)
        all_distance = all_distance + distance
    return all_distance / len(merged_coordinates)

def calculate_speed_inkmps(feature_distance, GSD, time_difference):
    distance = feature_distance * GSD / 100000
    speed = distance / time_difference
    return speed

i = 1
lastPictureTime = 0
while datetime.now().timestamp() - starttime < duration:
    
    # get new picture every 15 seconds
    if lastPictureTime == 0 or datetime.now().timestamp() - lastPictureTime >= 15:
        logger.debug("Take a new shot " + f'gps_image{i:02d}.jpg')
        cam.take_photo(f'gps_image{i:02d}.jpg', gps_coordinates=get_gps_coordinates(iss))
        i += 1
        lastPictureTime = datetime.now().timestamp()    
       
time_difference = get_time_difference('gps_image01.jpg','gps_image02.jpg')
image_1_cv, image_2_cv = convert_to_cv('gps_image01.jpg','gps_image02.jpg')
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculate_features(image_1_cv, image_2_cv, 1000)
matches = calculate_matches(descriptors_1, descriptors_2)

display_matches(image_1_cv, keypoints_1, image_2_cv, keypoints_2, matches)

coordinates_1, coordinates_2 = find_matching_coordinates(keypoints_1, keypoints_2, matches)
average_feature_distance = calculate_mean_distance(coordinates_1, coordinates_2)
speed = calculate_speed_inkmps(average_feature_distance, 12648, time_difference)
print(speed)
