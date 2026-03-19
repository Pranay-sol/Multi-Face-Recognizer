import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_num_classes(df):
    return len(df['Name'].unique())

def build_face_classifier(num_classes):
    # 1. Load the 'Body' (Pre-trained on ImageNet)
    # include_top=False removes the 1000-class layer
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # 2. FREEZE the body (Don't let the training change these layers)
    base_model.trainable = False
    
    # 3. Add the 'New Head'
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Flattens the 3D features into a 1D vector
    x = Dense(512, activation='relu')(x) # Hidden layer for complex patterns
    x = Dropout(0.5)(x) # Prevents memorizing (overfitting)
    
    # The final layer: size must match your number of names
    predictions = Dense(num_classes, activation='softmax')(x) 

    # 4. Construct the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return base_model,model

def fine_tune_model(base_model,model):
    base_model.trainable = True

    fine_tune_at = 100

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=0.00001), # 10x smaller than usual
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    return model

def model_training():
    data = pd.read_pickle("./face_master_data.pkl")
    encoder = LabelEncoder()
    data['Name'] = encoder.fit_transform(data['Name'])
    n_class = int(data['Name'].max()) +1

    train_x, test_x, train_y, test_y = train_test_split(data["Id"], data["Name"], test_size= 0.1)
    train_x = np.stack(train_x).astype('float32')
    train_y = np.array(train_y).astype('int32')

    base_model,model = build_face_classifier(n_class)

    model.fit(train_x, train_y, epochs=10, batch_size=32, validation_split=0.2)

    model = fine_tune_model(base_model, model)
    
    model.fit(train_x, train_y, epochs=15, batch_size=32, validation_split=0.2)


    model.save("face_recognition_model.h5")
    mapping_df = pd.DataFrame(encoder.classes_, columns=['Name'])
    mapping_df.to_csv("label_mapping.csv", index=True)    
    
    return model


if __name__ == "__main__":
    model_training()