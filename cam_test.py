import cv2
import logging
import os
import pandas as pd
import numpy as np
import urllib.request

def live_face_capture_id(name :str):
    #Load the Face Detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    logging.warning(f"Starting collection for: {name}. Press 's' to save a frame, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for the detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw a rectangle on the display frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame,f"Name : {name}",(x, y+h+10),cv2.FONT_HERSHEY_SIMPLEX,0.2,(255, 0, 0),1)
            # Crop the face
            face_roi = frame[y:y+h, x:x+w]
        
        window_name = "Press Q to quit, S to save" 
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Press 's' to save the cropped face
        if key == ord('s') and len(faces) > 0:
            # Resize to a standard CNN input size (e.g., 224x224)
            face_resized = cv2.resize(face_roi, (224, 224))
            
            img_name = f"./temp_images/{name}.jpg"
            cv2.imwrite(img_name, face_resized)
            logging.warning(f"Saved: {img_name}")
            break

        # Standard exit logic
        if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


# Now load the network

def image_face_capture_id(directory_path,output_dir, master_data: bool,target_size=(224, 224), confidence_threshold=0.5):
    face_matrix = []
    name_list = []

    config_path = "./DNN files/deploy.prototxt"
    model_path = "./DNN files/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    for filename in os.listdir(directory_path):
        logging.warning(f"File processing: {filename}")
        img_path = os.path.join(directory_path, filename)
        image = cv2.imread(img_path)
        if image is None: 
            logging.warning("No Image found.")
            continue
        
        annotated_img = image.copy()
        (h, w) = image.shape[:2]
        
        # 2. Convert image to a 'blob' for the detector
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # 3. Loop over the detections
        logging.warning(f"No. of faces detected: {detections.shape[2]}")
        face_count = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                face_count += 1
                logging.warning(f"{face_count} faces processing.")
                # Get coordinates for the face bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # --- STEP 1: Draw the Square (Bounding Box) ---
                # Color is BGR: (0, 255, 0) is Green. Thickness is 2 pixels.
                cv2.rectangle(annotated_img, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(annotated_img,f'{face_count}',(startX,startY - 10),cv2.FONT_HERSHEY_SIMPLEX,4,(0, 255, 0),3)
                # --- STEP 2: Crop and Prep for CNN ---
                face_crop = image[max(0, startY):min(h, endY), max(0, startX):min(w, endX)]
                if face_crop.size > 0:
                    # 1. Display the cropped face so you can see who it is
                    cv2.imshow("Who is this?", face_crop)                
                    # 2. Crucial: waitKey(1) allows the GUI to refresh and show the image
                    cv2.waitKey(1)
                    # 3. Ask for the name in the Terminal
                    person_name = input("Enter the name for the detected face: ")
                    cv2.destroyWindow("Who is this?")
                    face_prep = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_prep = cv2.resize(face_prep, target_size)
                    face_matrix.append(face_prep.astype("float32") / 255.0)
                    name_list.append(person_name)

        # --- STEP 3: Save the image with squares ---
        if face_count > 0:
            save_path = os.path.join(output_dir, f"detected_{filename}")
            cv2.imwrite(save_path, annotated_img)
            logging.warning(f"Saved {filename} with {face_count} faces to {output_dir}")

    logging.warning("All files processed.")
    if master_data == True:                
        return (pd.DataFrame({"Name" :name_list,"Id" :face_matrix}),True)
    else:
        return (pd.DataFrame({"Id" :face_matrix}),False)
