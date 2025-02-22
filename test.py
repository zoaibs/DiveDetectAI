# from ultralytics import YOLO
# import cv2
# import numpy as np
# import os
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib

# def euclidean_distance(pt1, pt2):
#     return np.linalg.norm(np.array(pt1) - np.array(pt2))

# # Load YOLO model
# model = YOLO('yolov8n-pose.pt')

# # Define feature extraction function
# def extract_features(video_path, label=None):
#     video = cv2.VideoCapture(video_path)
#     previous_keypoints, previous_time = None, None
#     features = []
    
#     while video.isOpened():
#         success, frame = video.read()
#         if not success:
#             break
        
#         results = model.predict(frame, conf=0.3, show=False)
#         if results and len(results[0].keypoints.xy) > 0:
#             keypoints = results[0].keypoints.xy[0].tolist()
#             current_time = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Time in seconds
            
#             if previous_keypoints is not None:
#                 velocity = [(keypoints[i][1] - previous_keypoints[i][1]) / (current_time - previous_time) for i in range(len(keypoints))]
#                 acceleration = [(velocity[i] - ((previous_keypoints[i][1] - keypoints[i][1]) / (current_time - previous_time))) / (current_time - previous_time) for i in range(len(keypoints))]
#                 unnatural_fall = any(abs(a) > 9.8 for a in acceleration)
#                 shoulder, hip = keypoints[5], keypoints[11]
#                 torso_angle = np.arctan2(hip[1] - shoulder[1], hip[0] - shoulder[0]) * (180 / np.pi)
#                 exaggerated_posture = torso_angle < -45 or torso_angle > 45
#                 contact_detected = euclidean_distance(keypoints[5], keypoints[6]) < 50 or euclidean_distance(keypoints[11], keypoints[12]) < 50
#                 reaction_time = current_time - previous_time
                
#                 features.append([np.mean(velocity), np.mean(acceleration), torso_angle, contact_detected, reaction_time, int(label) if label is not None else None])
                
#             previous_keypoints, previous_time = keypoints, current_time
    
#     video.release()
#     return features

# # Extract features from training videos
# def create_dataset(folder, label):
#     data = []
#     for file in os.listdir(folder):
#         video_path = os.path.join(folder, file)
#         data.extend(extract_features(video_path, label))
#     return data

# # Prepare dataset
# flop_data = create_dataset('/train/flop', label=1)
# real_data = create_dataset('/train/foul', label=0)
# dataset = pd.DataFrame(flop_data + real_data, columns=['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time', 'label'])

# # Train model
# X = dataset.drop(columns=['label'])
# y = dataset['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train, y_train)
# joblib.dump(clf, 'flop_classifier.pkl')

# # Evaluate model
# preds = clf.predict(X_test)
# print(f'Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%')

# # Predict on test videos
# def predict_flop(video_path):
#     clf = joblib.load('flop_classifier.pkl')
#     features = extract_features(video_path)
#     df = pd.DataFrame(features, columns=['velocity', 'acceleration', 'torso_angle', 'contact', 'reaction_time'])
#     predictions = clf.predict(df)
#     return 'Flop' if np.mean(predictions) > 0.5 else 'Real Fall'

# # Test prediction
# for file in os.listdir('test'):
#     print(f'{file}: {predict_flop(os.path.join("test", file))}')
