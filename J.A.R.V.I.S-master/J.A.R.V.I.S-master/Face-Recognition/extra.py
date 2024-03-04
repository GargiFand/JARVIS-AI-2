import cv2
import os

# Set the path for storing captured faces
path = 'samples'

# Create the samples directory if it doesn't exist
if not os.path.exists(path):
    os.makedirs(path)

# Set up the face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Set up the camera (you can change the argument to 1 if you have an external camera)
cap = cv2.VideoCapture(0)

# Counter for labeling each captured face
face_id = 1

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Save the captured face to the 'samples' directory
        face_image = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(path, f"user.{face_id}.jpg"), face_image)

        # Increment the face_id for the next face
        face_id += 1

    # Display the frame
    cv2.imshow('Capture Faces', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close the window
cap.release()
cv2.destroyAllWindows()
