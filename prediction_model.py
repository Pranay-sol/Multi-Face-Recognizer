from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import logging

#Load the trained Encoder
encoder = load_model("face_encoder_v1")

#Load your Master Gallery (The Pickle file)
df_master = pd.read_pickle("face_data_master.pkl")

# Pre-calculate the "Database Fingerprints" 
# (You only do this once at startup)
database_embeddings = encoder.predict(np.stack(df_master['Id'].values))
database_names = df_master['Name'].values

def identify_face(face_matrix, threshold=0.6):
    """
    face_matrix: a single (224, 224, 3) normalized image
    """
    # 1. Expand dimensions to (1, 224, 224, 3) for the CNN
    face_input = np.expand_dims(face_matrix, axis=0)
    
    # 2. Generate the unique fingerprint
    query_embedding = encoder.predict(face_input)[0]
    
    # 3. Find the closest match in your gallery
    distances = [euclidean(query_embedding, db_emb) for db_emb in database_embeddings]
    min_dist_idx = np.argmin(distances)
    
    if distances[min_dist_idx] < threshold:
        return database_names[min_dist_idx], distances[min_dist_idx]
    else:
        return "Unknown", distances[min_dist_idx]
    

def face_detection(data_file_path :str, attendance_file_path: str):
    attendance = []
    pred_file = pd.read_pickle(file_path)
    # 2. Loop through crops and identify
    for crop in pred_file.Id:
        name, score = identify_face(crop)
        attendance.append(name)
        logging.warning(f"Detected: {name} (Distance: {score:.4f})")
    
    pd.to_csv(f"{attendance_file_path}/attendance_list.csv")

face_detection("./face_data.pkl", "./")