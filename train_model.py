"""
ML Model Training Script for Poaching Detection
Trains a classifier to distinguish between gunshots and animal sounds.
"""

import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "ML_training/data"
MODEL_OUTPUT = "ML_training/gunshot_animal_model.pkl"
SCALER_OUTPUT = "ML_training/scaler.pkl"

# Classes: 0 = animals, 1 = gunshots
CLASSES = {
    "animals": 0,
    "gunshots": 1
}


def extract_features(file_path, sr=22050):
    """
    Extract audio features from a file.
    Returns a feature vector combining multiple audio characteristics.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr, duration=4.0)
        
        # Handle very short or silent audio
        if len(y) < sr * 0.1:  # Less than 0.1 seconds
            return None
            
        features = []
        
        # 1. MFCCs (13 coefficients) - mean and std
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # 2. Spectral Centroid - brightness of sound
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(spectral_centroid))
        features.append(np.std(spectral_centroid))
        
        # 3. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(spectral_bandwidth))
        features.append(np.std(spectral_bandwidth))
        
        # 4. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # 5. Zero Crossing Rate - noisiness
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 6. RMS Energy - loudness
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # 7. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spectral_contrast, axis=1))
        
        # 8. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_dataset():
    """Load and process all audio files from the dataset."""
    X = []
    y = []
    file_paths = []
    
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(DATA_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
            
        print(f"\nProcessing {class_name} (label={label})...")
        
        # Walk through all subdirectories
        file_count = 0
        for root, dirs, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    file_path = os.path.join(root, file)
                    features = extract_features(file_path)
                    
                    if features is not None:
                        X.append(features)
                        y.append(label)
                        file_paths.append(file_path)
                        file_count += 1
                        
                        if file_count % 50 == 0:
                            print(f"  Processed {file_count} files...")
        
        print(f"  Total {class_name}: {file_count} samples")
    
    return np.array(X), np.array(y), file_paths


def train_model():
    """Train and evaluate the ML model."""
    print("=" * 60)
    print("TRAINING ML MODEL FOR POACHING DETECTION")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/5] Loading and extracting features from audio files...")
    X, y, file_paths = load_dataset()
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dimensions: {X.shape[1] if len(X) > 0 else 'N/A'}")
    print(f"  Animals (0): {np.sum(y == 0)}")
    print(f"  Gunshots (1): {np.sum(y == 1)}")
    
    if len(X) < 10:
        print("ERROR: Not enough samples to train!")
        return
    
    # Split data
    print("\n[2/5] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    
    # Scale features
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models and compare
    print("\n[4/5] Training and comparing models...")
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        # Test score
        test_score = model.score(X_test_scaled, y_test)
        
        print(f"    CV Score: {cv_mean:.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"    Test Score: {test_score:.4f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print(f"\n  Best Model: {best_name} (Test Accuracy: {best_score:.4f})")
    
    # Detailed evaluation of best model
    print("\n[5/5] Evaluating best model...")
    y_pred = best_model.predict(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['animals', 'gunshots']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
    
    # Save model and scaler
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    joblib.dump(best_model, MODEL_OUTPUT)
    joblib.dump(scaler, SCALER_OUTPUT)
    
    print(f"  Model saved to: {MODEL_OUTPUT}")
    print(f"  Scaler saved to: {SCALER_OUTPUT}")
    
    # Test with poaching_alert.wav if it exists
    test_file = "poaching_alert.wav"
    if os.path.exists(test_file):
        print(f"\n  Testing with {test_file}...")
        test_features = extract_features(test_file)
        if test_features is not None:
            test_scaled = scaler.transform(test_features.reshape(1, -1))
            pred = best_model.predict(test_scaled)[0]
            proba = best_model.predict_proba(test_scaled)[0]
            class_name = 'gunshot' if pred == 1 else 'animal_distress'
            print(f"    Prediction: {class_name}")
            print(f"    Confidence: {max(proba)*100:.1f}%")
            print(f"    Probabilities: animal={proba[0]*100:.1f}%, gunshot={proba[1]*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return best_model, scaler


if __name__ == "__main__":
    train_model()
