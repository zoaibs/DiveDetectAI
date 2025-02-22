from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8n-pose model
model = YOLO('yolov8n-pose.pt')

# Open the video file
video = cv2.VideoCapture('test/roho_arnav_foul.mov')

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
out = cv2.VideoWriter('rafoul_skeleton.mp4',
                      cv2.VideoWriter_fourcc(*'XVID'),  # Use 'XVID' or 'H264'
                      fps,
                      (frame_width, frame_height))


# Process the video
while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    # Run YOLOv8 pose estimation on the frame
    results = model.predict(frame, conf=0.3, show=False)

    # Get the annotated frame
    annotated_frame = results[0].plot()
    
    # Write the frame
    out.write(annotated_frame)
    
    # Display the frame
    cv2.imshow("YOLOv8 Pose Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
video.release()
out.release()
cv2.destroyAllWindows()
