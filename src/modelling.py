from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

def create_generators(X_train, y_train, X_val, y_val, X_test, y_test, 
                     dropped_counts, look_back=14, batch_size=64):
    y_train_aligned = y_train.iloc[dropped_counts['train']:]
    y_val_aligned = y_val.iloc[dropped_counts['val']:]
    y_test_aligned = y_test.iloc[dropped_counts['test']:]
    
    train_gen = TimeseriesGenerator(
        data=X_train,
        targets=y_train_aligned,
        length=look_back,
        batch_size=batch_size,
        shuffle=False  
    )
    
    val_gen = TimeseriesGenerator(
        data=X_val,
        targets=y_val_aligned,
        length=look_back,
        batch_size=batch_size,
        shuffle=False
    )
    
    test_gen = TimeseriesGenerator(
        data=X_test,
        targets=y_test_aligned,
        length=look_back,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen, y_test_aligned


def build_model(look_back, n_features):
    #shape from processed features
    n_features = X_train.shape[1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(look_back, n_features))) 
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    model.compile(loss='mse', optimizer='adam')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        tf.keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
        ]   
    return model