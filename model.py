import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X = np.load('data/X.npy', allow_pickle=True)
    y = np.load('data/y.npy', allow_pickle=True)
    
    model = create_model(len(X[0]), len(y[0]))
    model.fit(X, y, epochs=200, batch_size=8, verbose=1)
    model.save('models/chatbot_model.h5')