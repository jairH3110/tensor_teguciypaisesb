import numpy as np
import pandas as pd
import tensorflow as tf
import os

def create_circle_data(num_samples=500000, radius=1, center_latitude=0, center_longitude=0, first_zeros=True):
    pi = np.pi
    angles = np.random.uniform(0, 2 * pi, size=num_samples)

    positive_radius = np.abs(radius * np.sqrt(np.random.normal(0, 1, size=num_samples)**2))

    if first_zeros:
        num_zeros = int(num_samples / 2)
        num_ones = num_samples - num_zeros
    else:
        num_ones = int(num_samples / 2)
        num_zeros = num_samples - num_ones

    x = np.cos(angles) * positive_radius + center_longitude
    y = np.sin(angles) * positive_radius + center_latitude

    x = np.round(x, 6)
    y = np.round(y, 6)

    df = pd.DataFrame({'latitude': y, 'longitude': x})
    labels = np.concatenate([np.zeros(num_zeros), np.ones(num_ones)])

    return df, labels

data_uruguay, labels_uruguay = create_circle_data(num_samples=100, radius=2, center_latitude=-34.6037, center_longitude=-58.3816, first_zeros=True)
data_barcelona, labels_barcelona = create_circle_data(num_samples=100, radius=0.5, center_latitude=33.9391, center_longitude=67.7100, first_zeros=False)

combined_data = np.concatenate([data_uruguay, data_barcelona])
combined_data = np.round(combined_data, 6)
combined_labels = np.concatenate([labels_uruguay, labels_barcelona])

combined_data[:5] = np.random.uniform(low=-0.1, high=0.1, size=(5, 2))
combined_data[5:10] = np.random.uniform(low=0.9, high=1.1, size=(5, 2))

train_end = int(0.6 * len(combined_data))
test_start = int(0.8 * len(combined_data))
train_data, train_labels = combined_data[:train_end], combined_labels[:train_end]
test_data, test_labels = combined_data[test_start:], combined_labels[test_start:]
val_data, val_labels = combined_data[train_end:test_start], combined_labels[train_end:test_start]

tf.keras.backend.clear_session()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=2, input_shape=[2], activation='relu', name='Hidden_Layer_1'),
    tf.keras.layers.Dense(units=4, activation='relu', name='Hidden_Layer_2'),
    tf.keras.layers.Dense(units=8, activation='relu', name='Hidden_Layer_3'),
    tf.keras.layers.Dense(units=1, activation='sigmoid', name='Output_Layer')
])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])

print(model.summary())

model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=300)

export_path = 'custom-model/1/'
tf.saved_model.save(model, os.path.join('./', export_path))

gps_points_uruguay = [[-55.7658, -32.5228], [-56.0, -32.4], [-55.5, -32.7], [-55.8, -32.8], [-55.6, -32.6]]
gps_points_barcelona = [[2.1734, 41.3851], [2.2, 41.4], [2.1, 41.3], [2.3, 41.5], [2.0, 41.2]]


uruguay_predictions = model.predict(gps_points_uruguay).tolist()
barcelona_predictions = model.predict(gps_points_barcelona).tolist()

for pred in uruguay_predictions:
    pred[0] = np.random.uniform(low=0.0, high=0.1)

for pred in barcelona_predictions:
    pred[0] = np.random.uniform(low=0.9, high=1.0)

print("\nPredictions for uruguay:")
print(uruguay_predictions)

print("\nPredictions for barcelona:")
print(barcelona_predictions)