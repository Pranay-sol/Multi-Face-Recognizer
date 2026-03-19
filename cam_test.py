import cv2
import logging
import os
import pandas as pd
import numpy as np

def image_from_dir(image_path,output_dir, master_data: bool):
    
    complete_df = pd.DataFrame(columns= ["Name","Id"])

    for file_name in os.listdir(image_path):
        full_path = os.path.join(image_path, file_name)
        logging.warning(full_path)
        # Read the image from the full path
        image = cv2.imread(full_path)    
            
        detections = identify_faces(image)
        df = convert_to_tensor(image, detections, master=True)
        complete_df = pd.concat([complete_df,df], axis= 0)
    return complete_df

def identify_faces(image):
    face_matrix = []
    name_list = []

    config_path = "./DNN files/deploy.prototxt"
    model_path = "./DNN files/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(config_path, model_path)

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    return detections

def convert_to_tensor(image,detections, master = False):
    (h, w) = image.shape[:2]

    face_matrix = []
    name_list = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        logging.warning(f"confidence:{confidence}")
        if confidence > 0.13:
            # Get coordinates for the face bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
                    
            # --- STEP 1: Draw the Square (Bounding Box) ---
            # Color is BGR: (0, 255, 0) is Green. Thickness is 2 pixels.
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            # --- STEP 2: Crop and Prep for CNN ---
            face_crop = image[max(0, startY):min(h, endY), max(0, startX):min(w, endX)]
            face_prep = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_prep = cv2.resize(face_prep, (224,224))
            face_matrix.append(face_prep.astype("float32") / 255.0)
                
            if face_crop.size > 0 and master == True:
                # 1. Display the cropped face so you can see who it is
                cv2.imshow("Who is this?", face_crop)                
                # 2. Crucial: waitKey(1) allows the GUI to refresh and show the image
                cv2.waitKey(1)
                # 3. Ask for the name in the Terminal
                person_name = input("Enter the name for the detected face: ")
                cv2.destroyWindow("Who is this?")
                name_list.append(person_name.lower())

    if master == True:        
        return pd.DataFrame({"Name" :name_list,"Id" :face_matrix})
    else:
        return pd.DataFrame({"Id" :face_matrix})