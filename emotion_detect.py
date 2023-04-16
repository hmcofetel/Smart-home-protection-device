import cv2
import numpy as np
import tensorflow as tf
from time import sleep
from picamera2 import Picamera2
# Load the pre-trained TensorFlow model for emotion detection
model = tf.keras.models.load_model('emotion_detection.h5')

# Define the emotion labels
EMOTIONS = ['Angry','Happy','Neutral','Sad','Surprise']

def detect_emotions():
    # Open the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (300, 300)}))
    picam2.start()

    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize the frame to 48x48 pixels
        resized = cv2.resize(gray, (48, 48), interpolation = cv2.INTER_AREA)
        
        # Convert the resized frame to a 4D tensor for feeding into the model
        tensor = np.expand_dims(resized, axis=-1)
        tensor = np.expand_dims(tensor, axis=0)
        tensor = tensor/255
        
        # Feed the tensor into the model to get a prediction
        prediction = model.predict(tensor)[0]
        
        # Get the index of the predicted emotion label
        label_index = np.argmax(prediction)
        
        # Get the corresponding emotion label
        label = EMOTIONS[label_index]
        
        # Draw the emotion label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Wait for a key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the camera and close the window
    cv2.destroyAllWindows()

# Call the function to start emotion detection
detect_emotions()