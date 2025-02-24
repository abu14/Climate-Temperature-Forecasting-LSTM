from src.data_processing import load_jena_data, clean_data, train_val_test_split
from src.features import FeatureEngineer, xy_splitter
from src.modeling import create_generators, build_model
import tensorflow as tf


if __name__ == "__main__":
    #load n split data
    df = load_jena_data()
    df = clean_data(df)
    (train_df, val_df, test_df), split_meta = train_val_test_split(df)
    
    #feature engineering
    engineer = FeatureEngineer()
    train_proc = engineer.fit_transform(train_df)
    val_proc = engineer.transform(val_df)
    test_proc = engineer.transform(test_df)
    
    # XY splitting with data alignment tracking
    (X_train, y_train), (X_val, y_val), (X_test, y_test), drop_meta = xy_splitter(
        train_proc, val_proc, test_proc
    )
    
    # Create generators
    train_gen, val_gen, test_gen, y_test_aligned = create_generators(
        X_train, y_train, 
        X_val, y_val,
        X_test, y_test,
        dropped_counts=drop_meta['dropped'],
        look_back=14,
        batch_size=64
    )

    model = build_model(look_back, X_train.shape[1])
    model.compile(loss='mse', optimizer='adam')
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.keras', save_best_only=True)
    ]
    
    history = model.fit(
        train_gen,
        epochs=25,
        validation_data=val_gen,
        callbacks=callbacks
    )