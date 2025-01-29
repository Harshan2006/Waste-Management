import numpy as np
from tensorflow.keras.models import load_model
from data import X_test, y_test, classes

# Load the model
model = load_model('model_save/recyclable_classifier_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Predict and display some sample results
predictions = model.predict(X_test)
for i in range(5):  # Display first 5 predictions
    true_label = classes[y_test[i]]
    predicted_label = classes[np.argmax(predictions[i])]
    print(f"True: {true_label}, Predicted: {predicted_label}")
