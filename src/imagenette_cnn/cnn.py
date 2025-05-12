# %%
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# %%
# Set memory growth to avoid allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth set to True for {device}")
else:
    print("No GPU found, using CPU")

# %%
# Set random seed for reproducibility
tf.random.set_seed(42)

# %%
# Use mixed precision to reduce memory usage (can speed up training on newer GPUs)
# Only enable if your GPU supports it
# try:
#     policy = tf.keras.mixed_precision.Policy("mixed_float16")
#     tf.keras.mixed_precision.set_global_policy(policy)
#     print("Using mixed precision policy")
# except:
#     print("Mixed precision not supported or enabled")

# %%
print("Loading Imagenette dataset...")
dataset, info = tfds.load("imagenette/160px", as_supervised=True, with_info=True)

# %%
num_classes = info.features["label"].num_classes
class_names = info.features["label"].names
train_ds = dataset["train"]
valid_ds = dataset["validation"]

# %%
# Target size for all images
TARGET_SIZE = (160, 160)


# %%
# Preprocess the data - including resizing to handle varying dimensions
def preprocess_data(image, label):
    # Resize images to consistent dimensions
    image = tf.image.resize(image, TARGET_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image, tf.one_hot(label, num_classes)


# %%
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
# train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

valid_ds = valid_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


# %%
def build_cnn_model():
    return models.Sequential(
        [
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(160, 160, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )


# %%
# Create and compile the model
print("Building and compiling the model...")
model = build_cnn_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# %%
# Save checkpoints only when validation improves (reduces disk I/O)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

# %%
# Implement early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# %%
# Reduce learning rate when plateauing
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1,
)

# %%
# Train the model
print("Training the model...")
epochs = 50

# %%
for images, _ in train_ds.take(1):
    print(f"Batch shape: {images.shape}")
    print(f"Memory footprint of batch: ~{images.numpy().nbytes / (1024 * 1024):.2f} MB")
    break

# %%
# Use TensorBoard for lightweight logging instead of storing all in memory
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir="./logs",
    histogram_freq=1,
    update_freq="epoch",
)

# %%
# Train with history stored but with memory-efficient callbacks
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint_callback,
    ],
    verbose=2,  # Less output to console
)

# %%
# Clean up to free memory
import gc

del model
gc.collect()
if physical_devices:
    tf.keras.backend.clear_session()
