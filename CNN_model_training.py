import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import accuracy_score, precision_score
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load your master file
df = pd.read_pickle("./face_data_master.pkl")

# 2. Group by name to make picking Positives/Negatives easier
names = df['Name'].unique()
face_groups = {name: np.stack(df[df['Name'] == name]['Id'].values) for name in names}

def get_triplet_batch(batch_size=32):
    anchors, positives, negatives = [], [], []
    
    for _ in range(batch_size):
        # Pick a random person for Anchor/Positive
        person = np.random.choice(names)
        # Pick a different person for Negative
        other_person = np.random.choice([n for n in names if n != person])
        
        # Select images
        idx_a, idx_p = np.random.choice(len(face_groups[person]), 2, replace=True)
        idx_n = np.random.choice(len(face_groups[other_person]))
        
        anchors.append(face_groups[person][idx_a])
        positives.append(face_groups[person][idx_p])
        negatives.append(face_groups[other_person][idx_n])
        
    return [np.array(anchors), np.array(positives), np.array(negatives)]

def create_base_cnn():
    model = models.Sequential([
        # Input Layer: Matches your matrix shape
        layers.Input(shape=(224, 224, 3)),
        
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        
        # The GAP Link: Converts spatial data to a vector
        layers.GlobalAveragePooling2D(),
        
        # The Embedding Layer: The final "Face Fingerprint"
        layers.Dense(128, activation=None), 
        
        # L2 Normalization: Makes Euclidean distance math work better
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])
    return model

base_cnn = create_base_cnn()

# Create 3 inputs
input_anchor = layers.Input(shape=(224, 224, 3))
input_positive = layers.Input(shape=(224, 224, 3))
input_negative = layers.Input(shape=(224, 224, 3))

# Link them to the SAME base_cnn
emb_a = base_cnn(input_anchor)
emb_p = base_cnn(input_positive)
emb_n = base_cnn(input_negative)

# The model now takes 3 images and outputs 3 embeddings
triplet_net = models.Model(
    inputs=[input_anchor, input_positive, input_negative], 
    outputs=[emb_a, emb_p, emb_n]
)

# Extract the batches (Anchor, Positive, Negative)
# Using the 'get_triplet_batch' logic from the previous step
X_anchor, X_pos, X_neg = get_triplet_batch(batch_size=32)

# Perform a training step
# The "y" (target) is usually just a dummy array of zeros because 
# the loss is calculated inside the custom Triplet Loss function.
triplet_net.train_on_batch([X_anchor, X_pos, X_neg], np.zeros((32,)))


def evaluate_embeddings(embeddings, labels, threshold=0.6):
    actuals = []
    predictions = []
    
    # Compare every pair in the validation set
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = euclidean(embeddings[i], embeddings[j])
            
            # 1 if they are actually the same person, 0 if different
            is_same = 1 if labels[i] == labels[j] else 0
            actuals.append(is_same)
            
            # 1 if the model THINKS they are the same (distance is low)
            pred_same = 1 if dist < threshold else 0
            predictions.append(pred_same)
            
    return accuracy_score(actuals, predictions)

#val_acc = evaluate_embeddings(val_embeddings, val_names)
#print(f"Verification Accuracy: {val_acc * 100:.2f}%")

#saving model for prediction
base_cnn.save("face_encoder_v1")