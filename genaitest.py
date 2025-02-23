import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import joblib
import json

# ---- Generative AI Setup ----
import google.generativeai as genai
genai.configure(api_key="AIzaSyAjwSb3D1loKnHrlb6NxzjcnC1nSHjOCCk")  # Replace with your actual API key
genai_model = genai.GenerativeModel("gemini-1.5-flash")

def getReason(verdict, weights):
    # verdict should be in lowercase: "foul" or "flop"
    if verdict == "foul":
        custom_prompt = (
            "I have data about a soccer player's joint movement. "
            "According to my ML random forest classifier with the following parameters: y velocity, acceleration, torso angle, contact, reaction time. "
            f"The feature differences are: {weights}. "
            "This player was fouled. "
            "Tell me the biggest reason there was a foul here. For example, a collision between the player's legs or nearly instant reaction time. "
            "Respond like a soccer referee without mentioning the model or the weights."
            "Also, dont talk about the torso angle."
        )
    else:
        custom_prompt = (
            "I have data about a soccer player's joint movement. "
            "According to my ML random forest classifier with the following parameters: y velocity, acceleration, torso angle, contact, reaction time. "
            f"The feature differences are: {weights}. "
            "This player flopped (was not actually fouled). "
            "Tell me the biggest reason there wasn't a foul here. For example, no contact between the player's legs or exaggerated fall timing. "
            "Respond like a soccer referee without mentioning the model or the weights."
            "Also, dont talk about the torso angle."
        )
    try:
        response = genai_model.generate_content(custom_prompt)
        if not response.text:
            return "No explanation provided."
        return response.text.strip()
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

# ---- Algorithm Functions ----

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Hardcoded baseline (normalized) values for features
baseline = {
    'velocity': 0.01912354192667923,
    'acceleration': 0.9464381441927009,
    'torso_angle': 0.034118353059798746,
    'contact': 0.0002936621262937371,
    'reaction_time': 2.6298694527314527e-05
}

# Set a threshold for prediction
THRESHOLD = 0.5

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
                            velocities = [(curr_skel[j][1] - prev_skel[j][1]) / dt for j in range(len(curr_skel))]
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
    Instead of hard 0/1 predictions, we use predict_proba to get probability estimates.
    Predictions are made for each skeleton (across frames) and then aggregated (by averaging)
    to yield a final probability. The final verdict is then computed based on a threshold.
    Additionally, we compute the normalized average feature vector for the video and then
    calculate the percent difference from the baseline for each feature.
    """
    features = extract_features(video_path)
    if not features:
        print("No features extracted from the video. Cannot make a prediction.")
        return None
    
    clf = joblib.load('flop_classifier.pkl')
    feature_columns = ['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time']
    df_features = pd.DataFrame(features, columns=feature_columns)
    
    probas = clf.predict_proba(df_features)
    avg_proba = np.mean(probas[:, 1])
    
    avg_feature_values = np.mean(df_features.values, axis=0)
    norm_feature_values = avg_feature_values / np.sum(avg_feature_values)
    
    percent_diff = {}
    for i, feature in enumerate(feature_columns):
        base_val = baseline[feature]
        current_val = norm_feature_values[i]
        percent_difference = ((current_val - base_val) / base_val) * 100
        percent_diff[feature] = percent_difference
    
    final_verdict = "Flop" if avg_proba > THRESHOLD else "Foul"
    return avg_proba, percent_diff, final_verdict

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
        annotated_frame = results[0].plot()
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
    valid_exts = ('.mp4', '.mov', '.avi')
    test_files = [f for f in os.listdir(test_folder)
                  if os.path.isfile(os.path.join(test_folder, f)) and f.lower().endswith(valid_exts)]
    test_files = sorted(set(test_files))
    
    for file in test_files:
        video_path = os.path.join(test_folder, file)
        result = predict_flop(video_path)
        if result is not None:
            avg_proba, percent_diff, verdict = result
            # Use generative AI to get the explanation
            # Convert verdict to lowercase for the prompt ("flop" vs "foul")
            reason = getReason(verdict.lower(), percent_diff)
            print(f"File: {file}, Prediction: {verdict}, Probability: {avg_proba:.2f}, Features: {percent_diff}")
            print(f"Reason: {reason}\n")
    
    if test_files:
        first_video = os.path.join(test_folder, test_files[0])
        print(f"Annotating skeletons for first test video: {first_video}")
        annotate_first_video(first_video, 'first_test_skeleton_output.mp4')

