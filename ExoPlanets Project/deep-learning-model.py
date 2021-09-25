import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split


def pipeline():

    koi_train = pd.read_csv('processed-data/koi_train.csv')
    X, y = koi_train.iloc[:, 0:24].to_numpy() , koi_train['koi_disposition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    n_features = X_train.shape[1]

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    model = tf.keras.Sequential()
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal', input_shape=(n_features, )))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(20, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0,  validation_split=0.3)
    loss = model.evaluate(X_train, y_train)


if __name__ == "__main__":
    pipeline()
