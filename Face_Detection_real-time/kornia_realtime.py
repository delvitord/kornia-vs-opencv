import torch
import kornia as K
import cv2
import time

# Load the face detection model from Kornia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detection_kornia = K.contrib.FaceDetector().to(device)

# Open the camera
cam = cv2.VideoCapture(0)
cam.set(3, 660)
cam.set(4, 500)

start_time = time.time()
frame_count = 0
total_detected_faces = 0
max_detection_distance_kornia = 0

while True:
    # Read a frame from the camera
    retV, frame = cam.read()

    # Convert the frame to tensor and move it to the specified device
    img_kornia = K.image_to_tensor(frame, keepdim=False).float().to(device)

    # Perform face detection using Kornia
    with torch.no_grad():
        dets_kornia = face_detection_kornia(img_kornia)
    
    # Update maximum detection distance in Kornia
    if len(dets_kornia[0]) > 0:
        max_detection_distance_kornia = max(
            max_detection_distance_kornia,
            max([(box[2] - box[0]) for box in dets_kornia[0]])
        )

    # Count the number of detected faces in the current frame
    total_detected_faces += len(dets_kornia[0])

    # Visualize the results for Kornia
    img_vis_kornia = frame.copy()
    for bbox in [box.int().tolist() for box in dets_kornia[0]]:
        x1, y1, x2, y2 = bbox[:4]
        img_vis_kornia = cv2.rectangle(img_vis_kornia, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with bounding boxes
    cv2.putText(img_vis_kornia, f"FPS: {frame_count / (time.time() - start_time):.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_vis_kornia, f"Total Detected Faces: {len(dets_kornia[0])}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Camera', img_vis_kornia)

    frame_count += 1

    # Check for user input to exit the loop
    close = cv2.waitKey(1) & 0xFF
    if close == 27 or close == ord('n'):
        break

# Release the camera
cam.release()
cv2.destroyAllWindows()