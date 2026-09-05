import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Environment Setup & GPU Configuration
# ---------------------------------------------------------
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found. Running on CPU.')
else:
    print('Found GPU at: {}'.format(device_name))

# Configure Matplotlib Backend
gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']
for gui in gui_env:
    try:
        matplotlib.use(gui, force=True)
        break
    except:
        continue

# ---------------------------------------------------------
# 2. Load Preprocessed Recurrence Plot Data
# ---------------------------------------------------------
# Update the file paths according to your environment
data_x_path = 'x_Apnea_Data_inception.pickle'
data_y_path = 'y_Apnea_Data_inception.pickle'

with open(data_x_path, 'rb') as f:
    X = pickle.load(f)

with open(data_y_path, 'rb') as g:
    y = pickle.load(g)

print("Loaded Data Shape:", X.shape)

# ---------------------------------------------------------
# 3. Data Normalization and Setup
# ---------------------------------------------------------
tf.random.set_seed(2)
np.random.seed(1)

IMG_SIZE = 299
NUM_CLASSES = 2

# Normalizing pixel values to [0, 1] range
X = X.astype('float32') / 255.0
y = np.array(y)

initializer = tf.keras.initializers.GlorotUniform(seed=7)
batch_size = 24
epochs = 200

# ---------------------------------------------------------
# 4. Stratified 5-Fold Cross Validation & Model Training
# ---------------------------------------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

j = 1
for train, test in kfold.split(X, y):
    print(f'\n--- Processing Fold {j} ---')
    
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    
    # Split Test set into Validation and Test sets
    X_test, X_valid, y_test, y_valid = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    # ---------------------------------------------------------
    # 5. Model Architecture (Transfer Learning with InceptionV3)
    # ---------------------------------------------------------
    base_model = tf.keras.applications.InceptionV3(
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        pooling='max'
    )
    
    x = base_model.output
    x = Dense(units=64, kernel_regularizer='l2', kernel_initializer=initializer, activation='relu')(x)
    x = Dense(units=1, kernel_regularizer='l2', kernel_initializer=initializer, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=x)
    
    opt = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.1, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(f'bestmodel_fold{j}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    ]
    
    # Train Model
    hist = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        callbacks=callbacks
    )
    
    # ---------------------------------------------------------
    # 6. Evaluation Metrics
    # ---------------------------------------------------------
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    yhat_probs = model.predict(X_test, verbose=0)[:, 0]
    yhat_classes = (yhat_probs > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, yhat_classes)
    precision = precision_score(y_test, yhat_classes)
    recall = recall_score(y_test, yhat_classes)
    tn, fp, fn, tp = confusion_matrix(y_test, yhat_classes).ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(y_test, yhat_classes)
    auc_score = roc_auc_score(y_test, yhat_probs)
    
    print(f"Fold {j} Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}")
    
    j += 1
