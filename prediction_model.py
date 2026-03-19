from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import logging
import cv2
from cam_test import identify_faces, convert_to_tensor
from scipy.optimize import linear_sum_assignment

#Load the trained model
final_model = load_model("face_recognition_model.h5")

def predict_face(model, image_matrix_df):
    name_idx = []
    predict_matrix = []
    for i in image_matrix_df:
        face_batch = np.expand_dims(i, axis=0)

        # 3. Predict
        prediction = model.predict(face_batch)

        # 4. Get the index of the highest score
        idx = np.argmax(prediction)
        name_idx.append(idx)
        predict_matrix.append(prediction)
    return name_idx, predict_matrix

def best_estimate_for_face(predictions):
    pred_array = np.array(predictions)
    
    pred_matrix = np.squeeze(pred_array)
    if pred_matrix.ndim == 1:
        pred_matrix = pred_matrix.reshape(1, -1)
    
    row_ind,name_indices = linear_sum_assignment(-pred_matrix)
    detected_names = get_names(name_indices)    
    return detected_names

def get_names(name_idx):
    mapping_df = pd.read_csv(r".\label_mapping.csv")
    name_lookup = mapping_df.set_index(mapping_df.columns[0])['Name'].to_dict()
    mapped_names = [name_lookup.get(i, "Unknown") for i in name_idx]
    
    return mapped_names

if __name__ == "__main__":
    image = cv2.imread(r".\temp_images\WhatsApp Image 2026-03-19 at 7.57.10 PM.jpeg")    
        
    detection = identify_faces(image)
    df = convert_to_tensor(image, detection)
    model = final_model
    x,predict = predict_face(model, df.iloc[:,0])
    a = best_estimate_for_face(predict)
    print(a)    
    