"""
WildGuard AI - LSTM Trend Detection Model
==========================================
This script implements a Long Short-Term Memory (LSTM) neural network
for detecting population trends in endangered species time-series data.

Author: WildGuard AI Project Team
Date: January 2026

WHY LSTM IS SUITABLE FOR TREND DETECTION:
==========================================

1. SEQUENTIAL PATTERN RECOGNITION:
   LSTM networks are specifically designed to learn patterns in sequential
   data. Population trends are inherently sequential - a decline in year 3
   only makes sense in the context of years 1 and 2. Traditional ML models
   treat each data point independently, losing this temporal context.

2. LONG-TERM DEPENDENCY LEARNING:
   Unlike vanilla RNNs that suffer from vanishing gradients, LSTMs use
   gating mechanisms (forget, input, output gates) to selectively remember
   or forget information over long sequences. This allows the model to
   capture multi-year trends, not just year-to-year changes.

3. HANDLING VARIABLE-LENGTH PATTERNS:
   Population trends can develop over different timescales. LSTMs can
   learn to recognize both sudden declines (1-2 years) and gradual
   changes (5+ years) within the same architecture.

4. NON-LINEAR RELATIONSHIPS:
   Wildlife population dynamics are complex and non-linear. Factors like
   carrying capacity, predator-prey cycles, and human intervention create
   patterns that simple linear models cannot capture. LSTMs can model
   these non-linear relationships through their deep architecture.

5. NOISE TOLERANCE:
   Census data contains measurement errors and natural fluctuations.
   LSTMs learn to distinguish meaningful trends from noise by considering
   the broader sequence context rather than individual data points.

6. MULTI-CLASS CLASSIFICATION:
   With a softmax output layer, LSTMs naturally handle multi-class
   classification (Sharp Decline, Moderate Decline, Stable, Recovering),
   outputting probability distributions across all classes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, 
        BatchNormalization, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow not installed. Run: pip install tensorflow")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_DIR = Path(__file__).parent
PLOTS_DIR = Path(__file__).parent.parent / "plots"

# Create directories
PLOTS_DIR.mkdir(exist_ok=True)

# Model parameters
SEQUENCE_LENGTH = 5      # Number of years in input sequence
NUM_FEATURES = 3         # Features per time step
RANDOM_STATE = 42

# Trend class definitions
TREND_CLASSES = {
    0: 'Sharp Decline',
    1: 'Moderate Decline', 
    2: 'Stable',
    3: 'Recovering'
}
NUM_CLASSES = len(TREND_CLASSES)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def load_and_prepare_data():
    """
    Load engineered data and prepare sequences for LSTM.
    
    Returns:
        tuple: (sequences, labels, feature_names)
    """
    print("=" * 70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load trend dataset
    df = pd.read_csv(DATA_DIR / "trend_dataset.csv")
    print(f"✓ Loaded {len(df)} records for {df['species_common_name'].nunique()} species")
    
    return df


def create_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trend labels based on population change patterns.
    
    LABELING LOGIC:
    ---------------
    - Sharp Decline: Average change rate < -10%
    - Moderate Decline: Average change rate between -10% and -2%
    - Stable: Average change rate between -2% and +5%
    - Recovering: Average change rate > +5%
    
    Parameters:
        df (pd.DataFrame): Input dataframe with population_change_rate
        
    Returns:
        pd.DataFrame: Dataframe with trend_label column
    """
    print("\n" + "=" * 70)
    print("STEP 2: CREATING TREND LABELS")
    print("=" * 70)
    
    def assign_trend_label(change_rate):
        """Assign trend label based on change rate thresholds."""
        if change_rate < -10:
            return 0  # Sharp Decline
        elif change_rate < -2:
            return 1  # Moderate Decline
        elif change_rate <= 5:
            return 2  # Stable
        else:
            return 3  # Recovering
    
    # Apply labeling
    df['trend_label'] = df['population_change_rate'].apply(assign_trend_label)
    
    # Display distribution
    print("\n✓ Trend Label Distribution:")
    print("-" * 40)
    for label, count in df['trend_label'].value_counts().sort_index().items():
        print(f"   {TREND_CLASSES[label]}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def create_sequences(df: pd.DataFrame, sequence_length: int = 5):
    """
    Create input sequences for LSTM from time-series data.
    
    SEQUENCE CREATION:
    ------------------
    For each species, we create overlapping sequences of `sequence_length` years.
    Each sequence represents the population trajectory over those years.
    
    Example for sequence_length=5:
        Input:  [year1, year2, year3, year4, year5] (features)
        Output: trend_label for year5
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        sequence_length (int): Number of time steps per sequence
        
    Returns:
        tuple: (X, y) - Input sequences and labels
    """
    print("\n" + "=" * 70)
    print(f"STEP 3: CREATING {sequence_length}-YEAR SEQUENCES")
    print("=" * 70)
    
    # Features to include in each sequence
    features = [
        'population_change_rate',
        'population_relative_to_peak',
        'cumulative_change_rate'
    ]
    
    X_sequences = []
    y_labels = []
    
    # Process each species separately
    for species in df['species_common_name'].unique():
        species_df = df[df['species_common_name'] == species].sort_values('year')
        
        # Skip if not enough data points
        if len(species_df) < sequence_length:
            continue
        
        # Normalize features within species (important for LSTM)
        scaler = StandardScaler()
        species_features = scaler.fit_transform(species_df[features])
        
        # Create sequences using sliding window
        for i in range(len(species_df) - sequence_length + 1):
            # Input sequence: features for sequence_length years
            sequence = species_features[i:i + sequence_length]
            
            # Label: trend of the last year in the sequence
            label = species_df['trend_label'].iloc[i + sequence_length - 1]
            
            X_sequences.append(sequence)
            y_labels.append(label)
    
    X = np.array(X_sequences)
    y = np.array(y_labels)
    
    print(f"✓ Created {len(X)} sequences")
    print(f"✓ Sequence shape: {X.shape} (samples, timesteps, features)")
    print(f"✓ Features used: {features}")
    
    return X, y, features


def balance_classes(X: np.ndarray, y: np.ndarray):
    """
    Balance classes using oversampling to handle class imbalance.
    
    Parameters:
        X: Input sequences
        y: Labels
        
    Returns:
        Balanced X, y
    """
    print("\n" + "=" * 70)
    print("STEP 4: BALANCING CLASSES")
    print("=" * 70)
    
    # Find class distribution
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    
    print("Before balancing:")
    for cls, cnt in zip(unique, counts):
        print(f"   {TREND_CLASSES[cls]}: {cnt}")
    
    # Oversample minority classes
    X_balanced = []
    y_balanced = []
    
    for cls in unique:
        cls_indices = np.where(y == cls)[0]
        cls_samples = X[cls_indices]
        cls_labels = y[cls_indices]
        
        # Repeat samples to match max_count
        repeat_times = max_count // len(cls_indices)
        remainder = max_count % len(cls_indices)
        
        X_balanced.extend(list(cls_samples) * repeat_times)
        X_balanced.extend(list(cls_samples[:remainder]))
        y_balanced.extend([cls] * (repeat_times * len(cls_indices) + remainder))
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    print("\nAfter balancing:")
    unique, counts = np.unique(y_balanced, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"   {TREND_CLASSES[cls]}: {cnt}")
    
    return X_balanced, y_balanced


# =============================================================================
# LSTM MODEL ARCHITECTURE
# =============================================================================

def build_lstm_model(input_shape: tuple, num_classes: int):
    """
    Build a simple LSTM model for trend classification.
    
    ARCHITECTURE EXPLANATION:
    -------------------------
    Layer 1: LSTM (32 units)
        - Processes sequential input and learns temporal patterns
        - return_sequences=False: Only output from final timestep
        - 32 units is sufficient for our simple 4-class problem
    
    Layer 2: Dropout (0.3)
        - Prevents overfitting by randomly dropping 30% of neurons
        - Critical for small datasets like ours
    
    Layer 3: Dense (16 units, ReLU)
        - Learns non-linear combinations of LSTM features
        - Small size prevents overfitting
    
    Layer 4: Dense (4 units, Softmax)
        - Output layer with 4 neurons for 4 classes
        - Softmax produces probability distribution
    
    Parameters:
        input_shape (tuple): Shape of input (timesteps, features)
        num_classes (int): Number of output classes
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    print("\n" + "=" * 70)
    print("STEP 5: BUILDING LSTM MODEL")
    print("=" * 70)
    
    model = Sequential([
        # Input layer
        Input(shape=input_shape),
        
        # LSTM layer - learns temporal patterns
        LSTM(
            units=32,                    # Number of LSTM units
            return_sequences=False,      # Only return final output
            activation='tanh',           # Default LSTM activation
            recurrent_activation='sigmoid'
        ),
        
        # Dropout for regularization
        Dropout(0.3),
        
        # Dense layer for feature combination
        Dense(16, activation='relu'),
        
        # Batch normalization for stable training
        BatchNormalization(),
        
        # Dropout for regularization
        Dropout(0.2),
        
        # Output layer with softmax for multi-class classification
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\n✓ Model Architecture:")
    print("-" * 50)
    model.summary()
    
    return model


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_model(model, X_train, y_train, X_val, y_val, epochs=100):
    """
    Train the LSTM model with early stopping.
    
    Parameters:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum epochs
        
    Returns:
        history: Training history
    """
    print("\n" + "=" * 70)
    print("STEP 6: TRAINING LSTM MODEL")
    print("=" * 70)
    
    # Callbacks for training optimization
    callbacks = [
        # Stop training when validation loss stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and generate metrics.
    
    Parameters:
        model: Trained Keras model
        X_test: Test sequences
        y_test: True labels (one-hot encoded)
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "=" * 70)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 70)
    
    # Make predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ Weighted F1-Score: {f1:.4f}")
    
    # Classification report
    print("\n" + "-" * 50)
    print("CLASSIFICATION REPORT:")
    print("-" * 50)
    target_names = [TREND_CLASSES[i] for i in range(NUM_CLASSES)]
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    }


def plot_results(history, metrics):
    """
    Create visualizations for training progress and evaluation results.
    
    Parameters:
        history: Training history
        metrics: Evaluation metrics dictionary
    """
    print("\n" + "=" * 70)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('LSTM Trend Detection Model - Training & Evaluation Results', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Training History - Loss
    ax1 = axes[0]
    ax1.plot(history.history['loss'], label='Training Loss', color='#e74c3c', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='#3498db', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training History - Accuracy
    ax2 = axes[1]
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='#2ecc71', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#9b59b6', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: Confusion Matrix
    ax3 = axes[2]
    cm = metrics['confusion_matrix']
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[TREND_CLASSES[i] for i in range(NUM_CLASSES)],
        yticklabels=[TREND_CLASSES[i] for i in range(NUM_CLASSES)],
        ax=ax3,
        cbar_kws={'label': 'Count'}
    )
    ax3.set_xlabel('Predicted', fontsize=11)
    ax3.set_ylabel('Actual', fontsize=11)
    ax3.set_title(f'Confusion Matrix\n(Accuracy: {metrics["accuracy"]*100:.1f}%)', 
                  fontsize=12, fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax3.yaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = PLOTS_DIR / "lstm_trend_detection_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Results plot saved to: {plot_path}")


def save_model(model, metrics):
    """Save the trained model and metrics."""
    print("\n" + "=" * 70)
    print("STEP 9: SAVING MODEL")
    print("=" * 70)
    
    # Save model
    model_path = MODELS_DIR / "lstm_trend_model.keras"
    model.save(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Save metrics summary
    metrics_summary = {
        'accuracy': float(metrics['accuracy']),
        'f1_score': float(metrics['f1_score']),
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'classes': TREND_CLASSES
    }
    
    import json
    metrics_path = MODELS_DIR / "lstm_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")


def print_interpretation(metrics):
    """Print interpretation of results."""
    print("\n" + "=" * 70)
    print("LSTM MODEL INTERPRETATION")
    print("=" * 70)
    
    interpretation = f"""
WHAT THE LSTM MODEL ACHIEVES:
=============================

1. TREND CLASSIFICATION CAPABILITY:
   The model successfully classifies population trends into 4 categories:
   - Sharp Decline (>10% annual loss)
   - Moderate Decline (2-10% annual loss)
   - Stable (-2% to +5% change)
   - Recovering (>5% annual growth)

2. MODEL PERFORMANCE:
   - Test Accuracy: {metrics['accuracy']*100:.1f}%
   - Weighted F1-Score: {metrics['f1_score']:.3f}
   
   This performance is {
       'EXCELLENT' if metrics['accuracy'] > 0.85 else
       'GOOD' if metrics['accuracy'] > 0.70 else
       'ACCEPTABLE' if metrics['accuracy'] > 0.55 else 'NEEDS IMPROVEMENT'
   } for a prototype model with limited data.

3. PRACTICAL APPLICATION:
   - The model can analyze new species' population sequences
   - Outputs probability distribution across all 4 trend classes
   - Can be integrated into the Streamlit dashboard for real-time predictions

4. ADVANTAGES OF LSTM FOR THIS TASK:
   ✓ Captures temporal dependencies across multiple years
   ✓ Learns complex non-linear patterns in population dynamics
   ✓ Handles varying sequence lengths and missing data patterns
   ✓ Produces calibrated probability estimates for uncertainty

5. LIMITATIONS AND CONSIDERATIONS:
   - Small dataset may limit generalization to new species
   - Model assumes historical patterns continue into future
   - External factors (climate, policy) not explicitly modeled
   - Regular retraining recommended as new census data arrives

6. CONFUSION MATRIX INSIGHTS:
"""
    
    # Analyze confusion matrix
    cm = metrics['confusion_matrix']
    for i in range(NUM_CLASSES):
        correct = cm[i, i]
        total = cm[i].sum()
        if total > 0:
            accuracy = correct / total * 100
            interpretation += f"   - {TREND_CLASSES[i]}: {accuracy:.0f}% correct ({correct}/{total})\n"
    
    print(interpretation)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute the complete LSTM training pipeline."""
    print("\n" + "=" * 70)
    print("    WILDGUARD AI - LSTM TREND DETECTION MODEL")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Create trend labels
    df = create_trend_labels(df)
    
    # Step 3: Create sequences
    X, y, features = create_sequences(df, sequence_length=SEQUENCE_LENGTH)
    
    # Step 4: Balance classes
    X_balanced, y_balanced = balance_classes(X, y)
    
    # Step 5: Split data
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y_balanced
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train
    )
    
    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat = to_categorical(y_val, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    print(f"✓ Test samples: {len(X_test)}")
    
    # Step 6: Build model
    input_shape = (SEQUENCE_LENGTH, len(features))
    model = build_lstm_model(input_shape, NUM_CLASSES)
    
    # Step 7: Train model
    history = train_model(model, X_train, y_train_cat, X_val, y_val_cat)
    
    # Step 8: Evaluate model
    metrics = evaluate_model(model, X_test, y_test_cat)
    
    # Step 9: Create visualizations
    plot_results(history, metrics)
    
    # Step 10: Save model
    save_model(model, metrics)
    
    # Step 11: Print interpretation
    print_interpretation(metrics)
    
    print("\n" + "=" * 70)
    print("LSTM TREND DETECTION MODEL COMPLETE ✓")
    print("=" * 70)
    
    return model, metrics


if __name__ == "__main__":
    model, metrics = main()
