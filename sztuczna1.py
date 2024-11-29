import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Krok 1: Przygotowanie danych
train_dir = 'data/train'  # Ścieżka do folderu z danymi treningowymi
test_dir = 'data/test1'   # Ścieżka do folderu z danymi testowymi

# Tworzenie generatorów danych z augmentacją
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Wczytywanie danych z podziałem na klasy
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

# Krok 2: Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=len(train_generator.class_indices), activation='softmax'))

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Krok 3: Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Krok 4: Wyświetlanie wyników trenowania
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Dokładność na przestrzeni epok')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Straty na przestrzeni epok')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

# Krok 5: Ocena modelu na danych testowych
loss, accuracy = model.evaluate(test_generator)
print(f'Accuracy on test data: {accuracy*100:.2f}%')

# Wyświetlanie mapy klas
print("Klasy:", train_generator.class_indices)

# Zapisanie modelu do pliku
model.save('dog_cat_classifier.h5')

# Krok 6: Przewidywanie na nowym obrazie
def predict_image(model_path, image_path):
    """
    Funkcja przewidująca klasę obrazu na podstawie wytrenowanego modelu.
    """
    model = load_model(model_path)
    class_indices = train_generator.class_indices
    class_labels = {v: k for k, v in class_indices.items()}  # Odwrócenie mapy klas
    
    # Wczytanie obrazu
    image = load_img(image_path, target_size=(64, 64))  # Dopasowanie rozmiaru obrazu
    image_array = img_to_array(image) / 255.0  # Normalizacja
    image_array = np.expand_dims(image_array, axis=0)  # Dodanie wymiaru batch
    
    # Przewidywanie
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    print(f'Przewidziana klasa: {class_labels[predicted_class]}')
    print(f'Pewność: {confidence*100:.2f}%')

# Testowanie funkcji przewidywania (zmień 'your_image.jpg' na własny obraz)
predict_image('dog_cat_classifier.h5', 'your_image.jpg')
