from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8n-pose model
model = YOLO('yolov8n-pose.pt')

# Open the video file
video = cv2.VideoCapture('test/foul5.mp4')

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
out = cv2.VideoWriter('foul5_skeleton.mp4',
                      cv2.VideoWriter_fourcc(*'XVID'),  # Use 'XVID' or 'H264'
                      fps,
                      (frame_width, frame_height))

# Process the video without printing frame-by-frame data
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Run YOLO pose estimation on the frame without verbose output
    results = model.predict(frame, conf=0.3, show=False, verbose=False)

    # Get the annotated frame using YOLO's built-in plotting method
    annotated_frame = results[0].plot()
    
    # Write the frame to the output video
    out.write(annotated_frame)
    
    # (Optional) You can comment out the display code to avoid terminal output:
    # cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
video.release()
out.release()
cv2.destroyAllWindows()
