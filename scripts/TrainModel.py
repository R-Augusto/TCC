# ===== TrainModel.py =====
import os, sys, contextlib

# ---------- SILENCIAR LOGS DE INÍCIO (absl/TF em stderr) ----------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=todos, 1=info, 2=warning, 3=error
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"  # corta logs verbosos do XLA

import tensorflow as tf
import pandas as pd

# ---------- MEMORY GROWTH ----------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print("Falha ao configurar memory growth:", e)

# ---------- MIXED PRECISION ----------
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision ativa:", mixed_precision.global_policy())

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from scripts.DataLoader import ImageLoader

# ---------- Configurações ----------
BATCH_SIZE   = 16
TARGET_SIZE  = (512, 512)
EPOCHS       = 30
INPUT_SHAPE  = (512, 512, 3)
MODEL_PATH   = "models/model_resnet50_binaria.keras"
IMAGE_DIR    = "data/images"

# ---------- Carregar dados ----------
train_df = pd.read_csv("results/train_labels.csv", names=["filename", "label"], header=None)
val_df   = pd.read_csv("results/val_labels.csv",   names=["filename", "label"], header=None)

train_loader = ImageLoader(IMAGE_DIR, train_df, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
val_loader   = ImageLoader(IMAGE_DIR, val_df,   batch_size=BATCH_SIZE, target_size=TARGET_SIZE)

# ---------- tf.data a partir do seu ImageLoader ----------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_loader.to_dataset(training=True,  shuffle_buffer=8192, drop_remainder=True, cache_in_ram=True)
val_ds   = val_loader.to_dataset(training=False, drop_remainder=True, cache_in_ram=True)

steps_per_epoch = len(train_df) // BATCH_SIZE
val_steps       = len(val_df)   // BATCH_SIZE

# ---------- Modelo ----------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1),                           # logits
    Activation('sigmoid', dtype='float32')  # saída em float32 p/ estabilidade
])

# ---------- Compile ----------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')],
)

# ---------- Callbacks ----------
callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
]

# ---------- Treino ----------
print("🚀 Iniciando o treinamento...")
with tf.device('/GPU:0'):
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

# ---------- Salvar histórico ----------
history_path = "results/history_resnet50.csv"
pd.DataFrame(history.history).to_csv(history_path, index=False)
print(f"✅ Histórico salvo em: {history_path}")
