import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import joblib

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Load YOLO pose estimation model (GPU will be used automatically if available)
yolo_model = YOLO('yolov8n-pose.pt')

def extract_features(video_path):
    """
    Extracts features from a test video using multi-skeleton detection.
    For each frame, each detected skeleton (if it has enough keypoints) is matched with its
    counterpart from the previous frame (by index), and its features (velocity, acceleration,
    torso angle, contact, reaction time) are computed individually.
    Returns a list of feature vectors.
    """
    video = cv2.VideoCapture(video_path)
    previous_keypoints = None
    previous_time = None
    features = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            results = yolo_model.predict(frame, conf=0.3, show=False, verbose=False)
            current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # Gather all skeletons (each must have at least 13 keypoints)
                current_skeletons = []
                for skeleton in results[0].keypoints.xy:
                    skel = skeleton.tolist()
                    if len(skel) < 13:
                        continue
                    current_skeletons.append(skel)
                
                # If previous frame data exists, compute features for each matching skeleton individually
                if previous_keypoints is not None and current_skeletons:
                    dt = current_time - previous_time
                    if dt > 0:
                        num_skel = min(len(previous_keypoints), len(current_skeletons))
                        for i in range(num_skel):
                            prev_skel = previous_keypoints[i]
                            curr_skel = current_skeletons[i]
                            # Compute per-keypoint vertical velocity (using y-coordinate)
                            velocities = [(curr_skel[j][1] - prev_skel[j][1]) / dt for j in range(len(curr_skel))]
                            # Compute a rough acceleration (using difference of velocities)
                            accelerations = [(velocities[j] - ((prev_skel[j][1] - curr_skel[j][1]) / dt)) / dt for j in range(len(curr_skel))]
                            shoulder = curr_skel[5]
                            hip = curr_skel[11]
                            torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
                            contact_detected = (euclidean_distance(curr_skel[5], curr_skel[6]) < 50 or
                                                euclidean_distance(curr_skel[11], curr_skel[12]) < 50)
                            reaction_time = dt
                            
                            feature_vector = [
                                np.mean(velocities),
                                np.mean(accelerations),
                                torso_angle,
                                int(contact_detected),
                                reaction_time
                            ]
                            features.append(feature_vector)
                
                previous_keypoints = current_skeletons
                previous_time = current_time
            pbar.update(1)
    
    video.release()
    return features

def predict_flop(video_path):
    """
    Extracts features from a test video and uses the trained classifier to predict the outcome.
    Predictions are made for each skeleton (across frames) and then aggregated (by averaging)
    to yield a final prediction score.
    """
    features = extract_features(video_path)
    if not features:
        print("No features extracted from the video. Cannot make a prediction.")
        return None
    
    clf = joblib.load('flop_classifier.pkl')
    
    # Convert features to a DataFrame with the same columns used during training
    feature_columns = ['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time']
    df_features = pd.DataFrame(features, columns=feature_columns)
    
    predictions = clf.predict(df_features)
    final_prediction = np.mean(predictions)
    return final_prediction

def annotate_first_video(video_path, output_video_path):
    """
    Annotates a video using YOLOv8's built-in plotting method (which displays all detected skeletons)
    and writes the annotated frames to an output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps,
                          (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = yolo_model.predict(frame, conf=0.3, show=False)
        annotated_frame = results[0].plot()  # This plots all detected skeletons
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved as {output_video_path}")

if __name__ == '__main__':
    test_folder = 'test'
    # Filter test files: process only files with typical video extensions
    valid_exts = ('.mp4', '.mov', '.avi')
    test_files = [f for f in os.listdir(test_folder)
                  if os.path.isfile(os.path.join(test_folder, f)) and f.lower().endswith(valid_exts)]
    test_files = sorted(set(test_files))  # Deduplicate and sort
    
    # Predict on each video in the test folder
    for file in test_files:
        video_path = os.path.join(test_folder, file)
        prediction = predict_flop(video_path)
        print(f'{file}: {prediction}')
    
    # Annotate the first video in the test folder
    if test_files:
        first_video = os.path.join(test_folder, test_files[0])
        print(f"Annotating skeletons for first test video: {first_video}")
        annotate_first_video(first_video, 'first_test_skeleton_output.mp4')
