import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Funktion zur Aufnahme von Bildern
def bilder_aufnehmen(personen_name, anzahl_bilder=1200):
    cap = cv2.VideoCapture(0)
    count = 0

    # Laden des Haar-Cascade-Modells für Gesichtserkennung
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    personen_ordner = f'/Users/benwiederhold/Desktop/Gesichtserkennung/training/{personen_name}'
    if not os.path.exists(personen_ordner):
        os.makedirs(personen_ordner)

    print(f"Starte Aufnahmen für {personen_name}. Bitte in die Kamera schauen.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Umwandlung des Bildes in Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Erkennen von Gesichtern
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Rahmen um das erkannte Gesicht zeichnen
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Ausschneiden des Gesichtsbereichs
            gesicht = frame[y:y+h, x:x+w]

        cv2.imshow('Bildaufnahme - Druecke "s" zum Speichern, "q" zum Beenden', frame)

        k = cv2.waitKey(1)

        if k & 0xFF == ord('s') and 'gesicht' in locals():
            bildpfad = os.path.join(personen_ordner, f"{personen_name}_{count}.jpg")
            cv2.imwrite(bildpfad, gesicht)
            print(f"Bild {count} gespeichert.")
            count += 1
            if count >= anzahl_bilder:
                break

        elif k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Interaktive Schleife zur Bildaufnahme für mehrere Personen
while True:
    personen_name = input("Geben Sie den Namen der Person ein (oder 'q' zum Trainiern): ")
    if personen_name == 'q':
        break
    bilder_aufnehmen(personen_name, anzahl_bilder=1200)

print("Bildaufnahme abgeschlossen. Beginne mit dem Training...")

# Trainings- und Validierungsdaten vorbereiten
train_dir = '/Users/benwiederhold/Desktop/Gesichtserkennung/training'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    validation_split=0.6)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training')

# Modell erstellen und trainieren
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10)

model.save('gesichtserkennungsmodell.h5')
