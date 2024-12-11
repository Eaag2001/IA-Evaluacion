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
    r'C:\Users\USUARIO DELL\Desktop\Inteligencia Artificial\Evaluacion\dataset\train',  # Asegúrate de que esta ruta es correcta
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

# Verificar las clases y el número de imágenes
print(f"Clases encontradas: {train_generator.class_indices}")
print(f"Número total de imágenes en el conjunto de entrenamiento: {train_generator.samples}")
print(f"Número total de imágenes en el conjunto de validación: {validation_generator.samples}")

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

# Ahora proceder con el entrenamiento
history = model.fit(
    train_generator,
    epochs=15,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluación del modelo en el conjunto de prueba (opcional)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/USUARIO DELL/Desktop/Inteligencia Artificial/Evaluacion/dataset/test',  # Ruta donde están las imágenes de prueba
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Evaluación del modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Precisión en el conjunto de prueba: {test_accuracy * 100:.2f}%')

# Ruta a la imagen para predecir (ajusta la ruta según la imagen que quieras predecir)
img_path = 'C:/Users/USUARIO DELL/Desktop/Inteligencia Artificial/Evaluacion/dataset/train/gatos/gato1.jpg'  # Ruta de la imagen

# Cargar la imagen con OpenCV
img = cv2.imread(img_path)

# Verificar si la imagen se cargó correctamente
if img is None:
    print(f"No se pudo cargar la imagen en la ruta: {img_path}")
else:
    # Convertir la imagen de BGR (OpenCV) a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar la imagen a 224x224
    img = cv2.resize(img, (224, 224))
    
    # Convertir la imagen a un arreglo NumPy y normalizarla
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir la dimensión del batch
    img_array /= 255.0  # Normalizar la imagen

    # Hacer la predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Obtener la clase con la mayor probabilidad

    # Mostrar el resultado
    class_labels = {0: 'Perro', 1: 'Gato', 2: 'Ave'}  # Etiquetas de las clases
    print(f'Predicción: {class_labels[predicted_class[0]]}')

# Guardar el modelo entrenado
model.save('web/modelo.keras')

