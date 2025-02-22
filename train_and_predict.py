import cv2
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import joblib

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Load YOLO pose estimation model
yolo_model = YOLO('yolov8n-pose.pt')

def extract_features(video_path):
    """
    Extract features from a test video. This function returns a list of feature vectors,
    each containing: [velocity, acceleration, torso_angle, contact, reaction_time].
    """
    video = cv2.VideoCapture(video_path)
    previous_keypoints, previous_time = None, None
    features = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            results = yolo_model.predict(frame, conf=0.3, show=False, verbose=False)
            if results and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].tolist()
                if len(keypoints) < 13:
                    pbar.update(1)
                    continue
                current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if previous_keypoints is not None and previous_time is not None and (current_time - previous_time) > 0:
                    velocity = [(keypoints[i][1] - previous_keypoints[i][1]) / (current_time - previous_time)
                                for i in range(len(keypoints))]
                    acceleration = [(velocity[i] - ((previous_keypoints[i][1] - keypoints[i][1]) / (current_time - previous_time)))
                                    / (current_time - previous_time) for i in range(len(keypoints))]
                    
                    shoulder = keypoints[5]
                    hip = keypoints[11]
                    torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
                    
                    contact_detected = (euclidean_distance(keypoints[5], keypoints[6]) < 50 or
                                        euclidean_distance(keypoints[11], keypoints[12]) < 50)
                    
                    reaction_time = current_time - previous_time
                    
                    features.append([np.mean(velocity), np.mean(acceleration), torso_angle, int(contact_detected), reaction_time])
                
                previous_keypoints, previous_time = keypoints, current_time
            pbar.update(1)
    
    video.release()
    return features

def predict_flop(video_path):
    """
    Extracts features from a test video and uses the trained classifier to predict the outcome.
    It aggregates per-frame predictions (here via a mean threshold) to return the final decision.
    """
    # Extract features from the test video
    features = extract_features(video_path)
    if not features:
        print("No features extracted from the video. Cannot make a prediction.")
        return None
    
    # Load the trained classifier
    clf = joblib.load('flop_classifier.pkl')
    
    # Make predictions for each feature vector
    predictions = clf.predict(features)
    
    # Aggregate predictions (for example, using a threshold on the mean prediction)
    final_prediction = np.mean(predictions) #'Flop' if np.mean(predictions) > 0.5 else 'Real Fall'
    return final_prediction

if __name__ == '__main__':
    # Process and predict for each video in the 'test' folder
    for file in os.listdir('test'):
        video_path = os.path.join('test', file)
        prediction = predict_flop(video_path)
        print(f'{file}: {prediction}')
