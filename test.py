import tensorflow as tf
import numpy as np
from tensorflow import keras

class AdversariallyHardenedCreditModel:
    def __init__(self, feature_count, epsilon=0.05):
        self.model = self._build_model(feature_count)
        self.epsilon = epsilon  # Perturbation size
        
    def _build_model(self, feature_count):
        """Build the credit decision neural network"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(feature_count,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _generate_adversarial_examples(self, x, y):
        """Generate adversarial examples using FGSM"""
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            predictions = self.model(x_tensor)
            loss = keras.losses.binary_crossentropy(y_tensor, predictions)
        
        gradients = tape.gradient(loss, x_tensor)
        signed_grad = tf.sign(gradients)
        # Users trying to get approval would subtract from gradient
        direction = tf.where(tf.equal(y_tensor, 0), -signed_grad, signed_grad)
        
        return (x_tensor + self.epsilon * direction).numpy()
    
    def adversarial_train(self, X_train, y_train, epochs=10, batch_size=64, validation_data=None):
        """Train the model with adversarial examples"""
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train on clean data
            self.model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
            
            # Generate adversarial examples
            X_adv = self._generate_adversarial_examples(X_train, y_train)
            
            # Train on adversarial examples
            self.model.fit(X_adv, y_train, epochs=1, batch_size=batch_size, verbose=0)
            
            # Evaluate
            if validation_data:
                metrics = self.model.evaluate(*validation_data, verbose=0)
                print(f"  Validation accuracy: {metrics[1]:.4f}")
    
    def predict(self, X):
        """Make predictions with the hardened model"""
        return self.model.predict(X)
    
    def evaluate_robustness(self, X_test, y_test, epsilon_range=[0, 0.05, 0.1]):
        """Evaluate model robustness at different perturbation levels"""
        results = []
        
        for eps in epsilon_range:
            if eps == 0:
                acc = self.model.evaluate(X_test, y_test, verbose=0)[1]
            else:
                # Save original epsilon
                original_epsilon = self.epsilon
                # Set new epsilon for test
                self.epsilon = eps
                # Generate adversarial examples
                X_adv = self._generate_adversarial_examples(X_test, y_test)
                # Reset epsilon
                self.epsilon = original_epsilon
                # Evaluate
                acc = self.model.evaluate(X_adv, y_test, verbose=0)[1]
            
            results.append(acc)
            print(f"Epsilon: {eps:.2f}, Accuracy: {acc:.4f}")
        
        return results
    
    def save(self, filepath):
        """Save the model"""
        self.model.save(filepath)


