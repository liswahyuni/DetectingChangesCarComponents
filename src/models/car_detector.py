import tensorflow as tf
import numpy as np
import cv2

class CarComponentDetector:
    def __init__(self):
        self.model = self._build_model()
        
    def _build_model(self):
        # Modified input shape to match common web camera/screenshot dimensions
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        
        # Lightweight feature extraction (using MobileNetV2-like architecture)
        x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Efficient blocks for real-time inference
        x = self._inverted_residual_block(x, 64, 1)
        x = self._inverted_residual_block(x, 128, 2)
        x = self._inverted_residual_block(x, 256, 2)
        
        # Simplified component status detection head
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(5, activation='sigmoid', name='status')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, train_data, train_labels, val_data, val_labels, epochs=30, batch_size=32):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_accuracy'
            )
        ]
        
        return self.model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

    def predict(self, image):
        # Preprocess image
        img = cv2.resize(image, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img, verbose=0)
        
        # Map predictions to component states with uppercase consistency
        components = {
            'Front Left Door': 'OPEN' if predictions[0][0] > 0.5 else 'CLOSED',
            'Front Right Door': 'OPEN' if predictions[0][1] > 0.5 else 'CLOSED',
            'Rear Left Door': 'OPEN' if predictions[0][2] > 0.5 else 'CLOSED',
            'Rear Right Door': 'OPEN' if predictions[0][3] > 0.5 else 'CLOSED',
            'Hood': 'OPEN' if predictions[0][4] > 0.5 else 'CLOSED'
        }
        
        return components

    def _inverted_residual_block(self, x, filters, stride):
        # Lightweight block for efficient inference
        expand = filters * 6
        
        x = tf.keras.layers.Conv2D(expand, (1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.DepthwiseConv2D((3, 3), stride, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        return x