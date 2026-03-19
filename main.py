import pandas as pd
import numpy as np
import logging
from cam_test import image_from_dir

def name_check_in_db(df):
    try:
        master = pd.read_pickle("./face_master_data.pkl")
        df_combine = pd.concat([master, df], axis = 0, ignore_index= True)
            
    except FileNotFoundError:
        df_combine = df.copy()
    
    df_combine.to_pickle("./face_master_data.pkl")
    return df

if __name__ == "__main__":
    x = image_from_dir( "./temp_images","./face_crop",master_data=True)
    _ = name_check_in_db(x)
#name_check_in_db(file_path = "./face_data.pkl",image_face_capture_id("./temp_images","./face_crop", master_data= False))

