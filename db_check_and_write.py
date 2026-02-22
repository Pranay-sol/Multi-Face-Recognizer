import pandas as pd
import numpy as np
import logging
from cam_test import live_face_capture_id, image_face_capture_id

# 1. Setup - Create a folder for your data

def name_check_in_db(file_path: str,op):
    new_file = op[0]
    master = a[1]
    try:
        df = pd.read_csv(file_path)
        logging.warning("Database Found, checking for duplicate.")
        if file_path == "./face_master_data.pkl":
            if name.lower() in df['Name'].str.lower():
                logging.warning('Duplicate Name.')
        df = pd.concat([df, a[0]], ignore_index= True)
            
    except:
        logging.warning("no pre-existing Database, creating new.")
        df = a[0]
        
    logging.warning(f"File sucessfully saved at {file_path}")
    return df.to_pickle(file_path)

name_check_in_db(file_path = "./face_master_data.pkl",image_face_capture_id("./temp_images","./face_crop",master_data=True))
name_check_in_db(file_path = "./face_data.pkl",image_face_capture_id("./temp_images","./face_crop", master_data= False))

