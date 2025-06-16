import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ========================
# Config & Reproducibility
# ========================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class ImprovedFusionModel:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None
        self.scalers = {}
        self.class_weights = None
        
    def load_and_preprocess_data(self, img_path, aud_path, label_path):
        """Load and preprocess the data with proper normalization."""
        print("Loading and preprocessing data...")
        
        # Load data
        X_img = np.load(img_path)
        X_aud = np.load(aud_path)
        y = np.load(label_path)
        
        print(f"Original data shapes - Image: {X_img.shape}, Audio: {X_aud.shape}, Labels: {y.shape}")
        
        # Check for and handle NaN/Inf values
        X_img = self._clean_data(X_img, "image")
        X_aud = self._clean_data(X_aud, "audio")
        
        # Normalize features separately for each modality
        self.scalers['image'] = StandardScaler()
        self.scalers['audio'] = StandardScaler()
        
        X_img_scaled = self.scalers['image'].fit_transform(X_img)
        X_aud_scaled = self.scalers['audio'].fit_transform(X_aud)
        
        print(f"After scaling - Image range: [{X_img_scaled.min():.3f}, {X_img_scaled.max():.3f}]")
        print(f"After scaling - Audio range: [{X_aud_scaled.min():.3f}, {X_aud_scaled.max():.3f}]")
        
        # Analyze class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        # Compute class weights for imbalanced datasets
        if len(unique) > 1:
            class_weights = compute_class_weight('balanced', classes=unique, y=y)
            self.class_weights = dict(zip(unique, class_weights))
            print(f"Class weights: {self.class_weights}")
        
        return X_img_scaled, X_aud_scaled, y
    
    def _clean_data(self, X, data_type):
        """Clean data by handling NaN and Inf values."""
        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values in {data_type} data. Replacing with median...")
            nan_mask = np.isnan(X)
            for col in range(X.shape[1]):
                if np.any(nan_mask[:, col]):
                    median_val = np.nanmedian(X[:, col])
                    X[nan_mask[:, col], col] = median_val
        
        # Check for Inf values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            print(f"⚠️  Found {inf_count} Inf values in {data_type} data. Clipping...")
            X = np.clip(X, -1e10, 1e10)
        
        return X
    
    def build_improved_model(self, img_dim, aud_dim, dropout_rate=0.3, l2_reg=0.001, fusion_strategy='concatenate'):
        """Build an improved fusion model with better architecture."""
        
        # Image processing branch
        img_input = Input(shape=(img_dim,), name="image_input")
        
        # Progressive dimensionality reduction for images
        x1 = Dense(512, activation="relu", kernel_regularizer=l2(l2_reg))(img_input)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(dropout_rate)(x1)
        
        x1 = Dense(256, activation="relu", kernel_regularizer=l2(l2_reg))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(dropout_rate)(x1)
        
        x1 = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(x1)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(dropout_rate * 0.5)(x1)  # Less dropout in final layers
        
        # Audio processing branch
        aud_input = Input(shape=(aud_dim,), name="audio_input")
        
        x2 = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(aud_input)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(dropout_rate)(x2)
        
        x2 = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x2)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(dropout_rate * 0.5)(x2)
        
        # Fusion layer
        if fusion_strategy == 'concatenate':
            fused = Concatenate(name="fusion_concat")([x1, x2])
        elif fusion_strategy == 'multiply':
            # Element-wise multiplication (requires same dimensions)
            x1_proj = Dense(64, activation="relu")(x1)
            x2_proj = Dense(64, activation="relu")(x2)
            fused = tf.keras.layers.Multiply()([x1_proj, x2_proj])
        else:
            fused = Concatenate()([x1, x2])
        
        # Classification head
        x = Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(fused)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate * 0.5)(x)
        
        x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = Dropout(dropout_rate * 0.3)(x)
        
        # Output layer
        output = Dense(1, activation="sigmoid", name="classification_output")(x)
        
        model = Model(inputs=[img_input, aud_input], outputs=output, name="improved_fusion_model")
        return model
    
    def train_with_cross_validation(self, X_img, X_aud, y, n_folds=5):
        """Train model with cross-validation for better evaluation."""
        print(f"Training with {n_folds}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_img, y)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            # Split data
            X_img_train, X_img_val = X_img[train_idx], X_img[val_idx]
            X_aud_train, X_aud_val = X_aud[train_idx], X_aud[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Build and compile model
            model = self.build_improved_model(X_img.shape[1], X_aud.shape[1])
            model.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss="binary_crossentropy",
                metrics=["accuracy", "precision", "recall"]
            )
            
            # Train model
            history = model.fit(
                [X_img_train, X_aud_train], y_train,
                validation_data=([X_img_val, X_aud_val], y_val),
                epochs=50,
                batch_size=32,
                class_weight=self.class_weights,
                callbacks=self._get_callbacks(fold),
                verbose=0
            )
            
            # Evaluate
            val_score = model.evaluate([X_img_val, X_aud_val], y_val, verbose=0)[1]  # accuracy
            cv_scores.append(val_score)
            print(f"Fold {fold + 1} validation accuracy: {val_score:.4f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        print(f"\nCross-validation results:")
        print(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        return cv_scores
    
    def train_final_model(self, X_img, X_aud, y, test_size=0.2):
        """Train the final model on all available data."""
        print("Training final model...")
        
        # Split data
        X_img_train, X_img_val, X_aud_train, X_aud_val, y_train, y_val = train_test_split(
            X_img, X_aud, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        # Build and compile model
        self.model = self.build_improved_model(X_img.shape[1], X_aud.shape[1])
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
        
        print(self.model.summary())
        
        # Train model
        history = self.model.fit(
            [X_img_train, X_aud_train], y_train,
            validation_data=([X_img_val, X_aud_val], y_val),
            epochs=100,
            batch_size=32,
            class_weight=self.class_weights,
            callbacks=self._get_callbacks("final"),
            verbose=1
        )
        
        # Final evaluation
        self._evaluate_model(X_img_val, X_aud_val, y_val)
        
        return history
    
    def _get_callbacks(self, fold_name):
        """Get training callbacks."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if fold_name == "final":
            callbacks.append(
                ModelCheckpoint(
                    filepath="model/fusion_model.pth",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def _evaluate_model(self, X_img_val, X_aud_val, y_val):
        """Evaluate the trained model."""
        print("\nFinal Model Evaluation:")
        
        # Predictions
        y_pred_prob = self.model.predict([X_img_val, X_aud_val])
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Classification report
        print(classification_report(y_val, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix:\n{cm}")
    
    def plot_training_history(self, history):
        """Plot training history with more metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Acc', color='blue')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Acc', color='red')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Train Precision', color='blue')
            axes[1, 0].plot(history.history['val_precision'], label='Val Precision', color='red')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train Recall', color='blue')
            axes[1, 1].plot(history.history['val_recall'], label='Val Recall', color='red')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_training_plots.png', dpi=300, bbox_inches='tight')
        plt.show()

# ========================
# Usage Example
# ========================
def main():
    
    # Initialize the improved model
    fusion_model = ImprovedFusionModel(seed=42)
    
    # Load and preprocess data
    X_img, X_aud, y = fusion_model.load_and_preprocess_data(
        "data/processed/X_image.npy",
        "data/processed/X_audio.npy",
        "data/processed/y.npy"
    )