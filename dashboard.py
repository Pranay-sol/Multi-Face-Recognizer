import streamlit as st
import pandas as pd
from cam_test import convert_to_tensor, identify_faces
from prediction_model import predict_face, best_estimate_for_face
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(layout="wide")
st.title("AI Attendance Marking")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

model = load_model("face_recognition_model.h5")
        
if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    col_photo, col_results = st.columns([1, 1])
    with col_photo:
        st.subheader("Input Image")
        st.image(image, caption="Uploaded Image", width=400)
    with col_results:
        st.subheader("Process & Results")
        if st.button("Detect Faces"):
            status = st.empty()
            status.info("Running detection...")
            
            cv2_image = np.array(image)
            detection = identify_faces(cv2_image)
            df = convert_to_tensor(cv2_image, detection)
            _,pred = predict_face(model, df.iloc[:,0])
            a = best_estimate_for_face(pred)
            
            status.empty()
            
            st.divider() # Simple line separator
            if a:
                st.success(f"Detected {len(a)} faces:")
                
                # --- This creates the nice vertical list of names ---
                for i, name in enumerate(a):
                    st.markdown(f"**Student {i+1}:** {name}")
                    # You can add a small spacer to prevent crowding
                    st.write("") 

                download_df = pd.DataFrame({"Name": a})
    
                # Convert DataFrame to CSV string
                csv_data = download_df.to_csv(index=False).encode('utf-8')

                # 3. Add the Download Button
                st.write("---") # Visual separator
                st.download_button(
                    label="📥 Download Names as CSV",
                    data=csv_data,
                    file_name="detected_faces.csv",
                    mime="text/csv",
                    help="Click to save the list of recognized names to your computer."
                )
            else:
                st.warning("No faces recognized in the image.")
                