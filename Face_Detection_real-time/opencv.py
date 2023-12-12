import cv2
import time

cam = cv2.VideoCapture(0)
cam.set(3, 660)
cam.set(4, 500)
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

start_time = time.time()
frame_count = 0
total_detected_faces = 0  # New metric for total detected faces

while True:
    retV, frame = cam.read()
    warna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adjust the scaleFactor and minNeighbors for better face detection accuracy
    faces = faceDetector.detectMultiScale(warna, scaleFactor=1.1, minNeighbors=5)

    # Reset the total_detected_faces for the current frame
    total_detected_faces = 0

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 63), 2)
        rec_muka = warna[y: y+h, x: x + w]
        total_detected_faces += 1

    cv2.putText(frame, f"FPS: {frame_count / (time.time() - start_time):.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Detected Faces: {total_detected_faces}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Camera', frame)
    frame_count += 1

    close = cv2.waitKey(1) & 0xFF
    if close == 27 or close == ord('n'):
        break

cam.release()
cv2.destroyAllWindows()
