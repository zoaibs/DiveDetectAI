import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import torch
from ultralytics import YOLO

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

model = YOLO('yolov8n-pose.pt')

def extract_features(video_path, label=None):
    """
    Extracts features from a video using multi-skeleton detection.
    For each frame, each detected skeleton (if it has enough keypoints) is matched with its
    counterpart from the previous frame (by index), and its features (velocity, acceleration,
    torso angle, contact, reaction time) are computed individually.
    Each skeleton's feature vector is then appended to the output.
    """
    video = cv2.VideoCapture(video_path)
    previous_keypoints = None  # List of keypoints for each detected skeleton from previous frame
    previous_time = None
    features = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Run YOLO pose estimation on the frame
            results = model.predict(frame, conf=0.3, show=False, verbose=False)
            current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # time in seconds
            
            if results and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # Gather all skeletons (each must have at least 13 keypoints)
                current_skeletons = []
                for skeleton in results[0].keypoints.xy:
                    skel = skeleton.tolist()
                    if len(skel) < 13:
                        continue
                    current_skeletons.append(skel)
                
                # If previous frame data exists, compute features for each skeleton that can be matched
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
                            # Compute torso angle using keypoints 5 (shoulder) and 11 (hip)
                            shoulder = curr_skel[5]
                            hip = curr_skel[11]
                            torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
                            # Detect contact (if adjacent keypoints are too close)
                            contact_detected = (euclidean_distance(curr_skel[5], curr_skel[6]) < 50 or
                                                euclidean_distance(curr_skel[11], curr_skel[12]) < 50)
                            reaction_time = dt
                            
                            # Compute average values over keypoints for velocity and acceleration
                            feature_vector = [
                                np.mean(velocities),
                                np.mean(accelerations),
                                torso_angle,
                                int(contact_detected),
                                reaction_time
                            ]
                            # Append a row: video_path, feature_vector, and label (if provided)
                            features.append([video_path] + feature_vector + ([label] if label is not None else []))
                
                # Update previous frame's skeletons and time
                previous_keypoints = current_skeletons
                previous_time = current_time
            pbar.update(1)
    
    video.release()
    return features

def create_dataset(folder, label, processed_files):
    data = []
    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)
        if video_path in processed_files:
            print(f"Skipping already processed file: {video_path}")
            continue
        file_features = extract_features(video_path, label)
        data.extend(file_features)
    return data

def annotate_first_video(video_path, output_video_path):
    """
    Annotates a video using YOLOv8's built-in plotting method (which shows all detected skeletons)
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
        # Run pose estimation and use YOLO's built-in plotting method
        results = model.predict(frame, conf=0.3, show=False)
        annotated_frame = results[0].plot()  # This plots all skeletons in the frame
        out.write(annotated_frame)
        cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved as {output_video_path}")

def main():
    csv_file = 'flop_dataset.csv'
    # Load existing dataset if available
    if os.path.exists(csv_file):
        existing_dataset = pd.read_csv(csv_file)
        if 'video_path' in existing_dataset.columns:
            processed_files = set(existing_dataset['video_path'].tolist())
        else:
            processed_files = set()
    else:
        processed_files = set()
        existing_dataset = pd.DataFrame(columns=['video_path', 'velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time', 'label'])
    
    # Process training data from both classes
    flop_data = create_dataset('train/flop', label=1, processed_files=processed_files)
    real_data = create_dataset('train/foul', label=0, processed_files=processed_files)
    
    new_data = flop_data + real_data
    if new_data:
        new_df = pd.DataFrame(new_data, columns=['video_path', 'velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time', 'label'])
        combined_dataset = pd.concat([existing_dataset, new_df], ignore_index=True)
    else:
        combined_dataset = existing_dataset
    
    combined_dataset.to_csv(csv_file, index=False)
    print(f"Feature extraction complete. Data saved to {csv_file}.")
    
    # --- Train the classifier ---
    # Load the dataset and drop the 'video_path' column so that only numeric features are used
    dataset = pd.read_csv(csv_file)
    feature_columns = ['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time']
    X = dataset[feature_columns]
    y = dataset['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print(f"Validation Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    
    joblib.dump(clf, 'flop_classifier.pkl')
    print("Model training complete. Saved as flop_classifier.pkl.")
    
    # --- Annotate the first training video from the 'train/flop' folder ---
    train_flop_folder = 'train/flop'
    first_video = None
    for file in os.listdir(train_flop_folder):
        first_video = os.path.join(train_flop_folder, file)
        break
    if first_video:
        print(f"Annotating skeletons for first training video: {first_video}")
        annotate_first_video(first_video, 'first_skeleton_output.mp4')

if __name__ == '__main__':
    main()
