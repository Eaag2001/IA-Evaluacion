import numpy as np
import tensorflow as tf
import cv2  # OpenCV para cargar imágenes
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Configuración para cargar y normalizar las imágenes, con separación para validación
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 80% entrenamiento, 20% validación

train_generator = train_datagen.flow_from_directory(
    r'C:\Users\USUARIO DELL\Desktop\Inteligencia Artificial\Evaluacion\dataset\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    r'C:\Users\USUARIO DELL\Desktop\Inteligencia Artificial\Evaluacion\dataset\train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Cargar el modelo base (VGG16 preentrenado) sin las capas finales (incluye las características)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # No entrenar las capas de VGG16, solo utilizarlas como extractor de características

# Crear el modelo con las capas de VGG16 y nuestras propias capas finales
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Número de clases basado en tu dataset
])

# Compilar el modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    epochs=15,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluación del modelo en el conjunto de prueba
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/USUARIO DELL/Desktop/Inteligencia Artificial/Evaluacion/dataset/test',  # Ruta donde están las imágenes de prueba
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')

# Inicia la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Captura el fotograma de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar el fotograma en la ventana de la cámara
    cv2.imshow("Prediccion", frame)

    # Esperar la tecla para tomar la foto
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Si se presiona la tecla "s" para tomar la foto
        # Guardar la imagen
        img_path = 'captured_image.png'
        cv2.imwrite(img_path, frame)

        # Convertir la imagen a RGB y redimensionar para la predicción
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))

        # Convertir la imagen a un arreglo NumPy y normalizarla
        img_array = img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión del batch
        img_array /= 255.0  # Normalizar la imagen

        # Realizar la predicción
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)

        # Mostrar el resultado
        class_labels = {0: 'Perro', 1: 'Gato', 2: 'Ave'}
        print(f'Prediccion: {class_labels[predicted_class[0]]}')

        # Mostrar el resultado en la ventana de la cámara
        cv2.putText(frame, f'Prediccion: {class_labels[predicted_class[0]]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar la imagen tomada con la predicción
        cv2.imshow('Imagen capturada', frame)

    elif key == ord('q'):  # Si se presiona la tecla "q", salir del bucle
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
