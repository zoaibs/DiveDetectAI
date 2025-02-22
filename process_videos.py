from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Load YOLO model
model = YOLO('yolov8n-pose.pt')

# Feature extraction function
def extract_features(video_path, label=None):
    video = cv2.VideoCapture(video_path)
    previous_keypoints, previous_time = None, None
    features = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames for progress bar

    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break

            results = model.predict(frame, conf=0.3, show=False, verbose=False)  # Suppress logs

            if results and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].tolist()
                if len(keypoints) < 12:  # Ensure we have at least 12 keypoints (to access index 5 and 11)
                    pbar.update(1)
                    continue  # Skip this frame if keypoints are incomplete

                current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Time in seconds

                if previous_keypoints is not None:
                    velocity = [(keypoints[i][1] - previous_keypoints[i][1]) / (current_time - previous_time)
                                for i in range(len(keypoints))]
                    acceleration = [(velocity[i] - ((previous_keypoints[i][1] - keypoints[i][1]) / (current_time - previous_time)))
                                    / (current_time - previous_time) for i in range(len(keypoints))]
                    
                    shoulder, hip = keypoints[5], keypoints[11]
                    torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
                    
                    contact_detected = (
                        euclidean_distance(keypoints[5], keypoints[6]) < 50 or
                        euclidean_distance(keypoints[11], keypoints[12]) < 50
                    )
                    reaction_time = current_time - previous_time
                    
                    features.append([np.mean(velocity), np.mean(acceleration), torso_angle, contact_detected, reaction_time, label])
                
                previous_keypoints, previous_time = keypoints, current_time

            pbar.update(1)  # Update progress bar

    video.release()
    return features

def create_dataset(folder, label, processed_files):
    data = []
    for file in os.listdir(folder):
        video_path = os.path.join(folder, file)
        # Skip the file if its path is already in the dataset.
        if video_path in processed_files:
            print(f"Skipping already processed file: {video_path}")
            continue
        file_features = extract_features(video_path, label)
        # Prepend the video path to each feature row so we know which file it came from.
        for feat in file_features:
            data.append([video_path] + feat)
    return data

def main():
    csv_file = 'flop_dataset.csv'
    # Load existing dataset if it exists
    if os.path.exists(csv_file):
        existing_dataset = pd.read_csv(csv_file)
        processed_files = set(existing_dataset['video_path'].tolist())
        print(f"Found existing dataset with {len(processed_files)} processed files.")
    else:
        processed_files = set()
        existing_dataset = pd.DataFrame(columns=['video_path', 'velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time', 'label'])

    # Process new training data for both classes
    flop_data = create_dataset('train/flop', label=1, processed_files=processed_files)
    real_data = create_dataset('train/foul', label=0, processed_files=processed_files)
    
    new_data = flop_data + real_data
    if new_data:
        new_df = pd.DataFrame(new_data, columns=['video_path', 'velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time', 'label'])
        combined_dataset = pd.concat([existing_dataset, new_df], ignore_index=True)
    else:
        combined_dataset = existing_dataset

    # Save the updated dataset
    combined_dataset.to_csv(csv_file, index=False)
    print(f"Feature extraction complete. Data saved to {csv_file}.")

if __name__ == '__main__':
    main()
