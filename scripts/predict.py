import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def predict_and_plot(model_path, test_gen, y_test_aligned):
    model = load_model(model_path)
    y_pred = model.predict(test_gen).flatten()
    
    # Your exact plotting code
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_aligned, label='Actual', color='blue')
    plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
    plt.title("Your Exact Title")
    plt.show()

if __name__ == "__main__":
    # (You'd load test_gen and y_test_aligned here)
    predict_and_plot('models/best_model.keras', test_gen, y_test_aligned)