import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Load the processed data
X = np.load('data/X.npy', allow_pickle=True)
y = np.load('data/y.npy', allow_pickle=True)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(X[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(X, y, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('models/chatbot_model.h5', hist)

print("Model created and saved.")
print(f"Final accuracy: {hist.history['accuracy'][-1]}")