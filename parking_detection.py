import os
import numpy as np
import cv2
import Mask_RCNN.mrcnn.config
import Mask_RCNN.mrcnn.utils
import Mask_RCNN.mrcnn.visualize
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from Mask_RCNN.mrcnn.model import MaskRCNN
from pathlib import Path
from sklearn.cluster import KMeans


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(Mask_RCNN.mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80
    DETECTION_MIN_CONFIDENCE = 0.7


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    Mask_RCNN.mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Create a Mask-RCNN model in inference mode
model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

# Load pre-trained model
model.load_weights(COCO_MODEL_PATH, by_name=True)

# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']

VIDEO_SOURCE = "videos/test_video.MOV"

# Load the video file we want to run detection on
video_capture = cv2.VideoCapture(VIDEO_SOURCE)
n_clusters = 2


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids, scores):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [3]:
            if scores[i] > 0.8:
                car_boxes.append(box)

    return np.array(car_boxes)


def create_mean_points(car_boxes):
    """
        Input: car_boxes - a list of parked cars locations in the format (y1, x1, y2, x2)
        Functionality: find avg point (x, y) based on (y1, x1, y2, x2)
        Output: mean_points - an np array of shape (len(car_boxes), 2), where each row is the
                              avg point (x, y)
    """
    mean_points = np.empty((len(car_boxes), 2))
    for i, box in enumerate(car_boxes):
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        mean_points[i][0] = x
        mean_points[i][1] = y

    return mean_points


def create_zones(car_boxes, n_clusters=n_clusters):
    """
        Input: car_boxes - a list of parked cars locations in the format (y1, x1, y2, x2)
               n_clusters - number of clusters to form
        Functionality: use k-means clustering to form {n_clusters} different zones
        Output: zones - a list of lists, each list containing cars locations in the format (y1, x1, y2, x2)
                        that are in a particular zone
                kmeans - the k-means model used to form the clusters
    """

    zones = [[] for _ in range(n_clusters)]
    mean_points = create_mean_points(car_boxes)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(mean_points)
    centroids = kmeans.cluster_centers_
    for i in range(len(centroids)):
        print("Zone {} centroid : {}".format(i, centroids[i]))

    predictions = kmeans.predict(mean_points)
    for i, num in enumerate(predictions):
        zones[num].append(car_boxes[i])

    return zones, kmeans


def find_area(zone):
    """
        Input: zone - a list containing the cars locations in the format (y1, x1, y2, x2)
        Functionality: find the total area covered by the box of each car in a zone
        Output: total_area of a zone
    """
    sorted_by_x = sorted(zone, key=lambda x: x[1])

    total_area = 0
    for i, box in enumerate(sorted_by_x):
        y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
        if i != 0:
            prev_box = sorted_by_x[i - 1]
            y1_prev, x1_prev, y2_prev, x2_prev = prev_box[0], prev_box[1], prev_box[2], prev_box[3]
            if x1_prev <= x1 <= x2_prev and y1_prev <= y1 <= y2_prev:
                intersection_width = abs(x2_prev - x1)
                intersection_height = abs(y2_prev - y1)
                total_area -= (intersection_width * intersection_height)

        total_area += abs(y2 - y1) * abs(x2 - x1)
    return total_area


def detect_open_space(pred_zone, zone, zone_area):
    """
        Input: pred_zone - a list containing the cars locations in the format (y1, x1, y2, x2)
                        that are CURRENTLY in the zone
               zone - a list containing the cars locations in the format (y1, x1, y2, x2)
                        that were in the ORIGINAL zone
                zone_area - area of the ORIGINAL zone
        Functionality: determine if there are open spaces for a car to park
        Output: a boolean indicating if there is open space for a car
    """
    area = find_area(pred_zone)
    if zone_area - area >= zone_area/len(zone):
        print("There is space in this zone")
        return True

    print("There is no space in this zone")
    return False


car_boxes = None
zones = None
zones_area = []
zones_free_space_frequency = [0 for i in range(n_clusters)]
k_means = None


# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imwrite("plot_training_frame.png", rgb_image)
    plt.imshow(mpimg.imread('plot_training_frame.png'))
    plt.savefig('plot_training_frame.png')

    # Run the image through the Mask R-CNN model to get results.
    results = model.detect([rgb_image], verbose=0)

    # Mask R-CNN assumes we are running detection on multiple images.
    # We only passed in one image to detect, so only grab the first result.
    r = results[0]

    # The r variable will now have the results of detection:
    # - r['rois'] are the bounding box of each detected object
    # - r['class_ids'] are the class id (type) of each detected object
    # - r['scores'] are the confidence scores for each detection
    # - r['masks'] are the object masks for each detected object (which gives you the object outline)

    # Filter the results to only grab the car / truck bounding boxes
    car_boxes = get_car_boxes(r['rois'], r['class_ids'], r['scores'])
    for box in car_boxes:
        y1, x1, y2, x2 = box

        # Draw the box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    cv2.imwrite("training_frame.png", frame)

    zones, k_means = create_zones(car_boxes)
    for i, zone in enumerate(zones):
        zones_area.append(find_area(zone))

    break

print("Zones acquired")
for i, zone in enumerate(zones):
    print("Zone {}: {}".format(i, zone))

NUMBER_FRAMES_PER_SEC = 30
num_secs = 2
frame_in_sec = 0
secs = 0

# Loop over each frame of video
while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    frame_in_sec += 1
    if frame_in_sec == NUMBER_FRAMES_PER_SEC * num_secs:
        frame_in_sec = 0
        secs += 2
    else:
        continue

    print("Number of seconds: ", secs)

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.detect([rgb_image], verbose=0)
    r = results[0]

    curr_car_boxes = get_car_boxes(r['rois'], r['class_ids'], r['scores'])
    for box in curr_car_boxes:
        y1, x1, y2, x2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    curr_mean_points = create_mean_points(curr_car_boxes)
    curr_predictions = k_means.predict(curr_mean_points)
    curr_zones = [[] for _ in range(n_clusters)]

    for i, num in enumerate(curr_predictions):
        curr_zones[num].append(curr_car_boxes[i])

    for i, zone in enumerate(curr_zones):
        if len(zone) < len(zones[i]):
            zones_free_space_frequency[i] += 1
            if zones_free_space_frequency[i] > 15:
                zones_free_space_frequency[i] = 0
                print("Zone Number ", i)
                print("New Zone: ", zone)
                print("Original Zone: ", zones[i])
                cv2.imwrite("frame_secs_{}.png".format(secs), frame)
                detect_open_space(curr_zones[i], zones[i], zones_area[i])

