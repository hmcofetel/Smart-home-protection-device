import face_recognition
import cv2
import numpy as np
import os
from picamera2 import Picamera2
from anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.src.generate_patches import CropImage
from anti_spoofing.src.utility import parse_model_name

RED = (0,0,255)
GREEN = (255,0,0)
BLUE = (0,255,0)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (600, 800)}))
picam2.start()


obama_image = face_recognition.load_image_file("./data_face/minhco.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]



known_face_encodings = [
    obama_face_encoding
]
known_face_names = [
    "Minh Co"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
model_test = AntiSpoofPredict(0)
def check(image, model_dir):
    global model_test
    image_cropper = CropImage()
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0
    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img,os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    value = prediction[0][label]/2

    return label, value


while True:
    
    frame = picam2.capture_array()
    try:
        print(check(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "./anti_spoofing/resources/anti_spoof_models"))
    except:
        pass
    rgb_small_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    if process_this_frame:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
       
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    face_names.append("Unknown")
    process_this_frame = not process_this_frame


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        color = GREEN
        if name != "Unknown":
            color = BLUE
        

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()