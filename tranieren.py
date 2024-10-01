from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os

# Funktion zum Löschen alter Trainingsdaten
def delete_old_training_data(files=['chatbot_model.h5', 'vectorizer.pkl', 'classes.pkl']):
    for file in files:
        if os.path.exists(file):
            os.remove(file)
            print(f"{file} gelöscht.")
        else:
            print(f"{file} nicht gefunden, kein Löschen erforderlich.")

delete_old_training_data()

# JSON-Daten laden
with open('intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()
ignore_letters = ['?', '!', '.', ',']

# Daten vorbereiten
documents = []
classes = []
all_sentences = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        word_list = [lemmatizer.lemmatize(w.lower()) for w in word_list if w not in ignore_letters]
        documents.append((word_list, intent['tag']))
        all_sentences.append(' '.join(word_list))  # Speichere den Satz als String
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

classes = sorted(set(classes))

# Datenaugmentierung: Ersetze einige Wörter durch Synonyme
def augment_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    augmented_sentence = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()  # Wähle das erste Synonym
            augmented_sentence.append(synonym)
        else:
            augmented_sentence.append(word)
    return ' '.join(augmented_sentence)

# Füge die augmentierten Sätze zu deinem Datensatz hinzu
augmented_sentences = []
for sentence in all_sentences:
    augmented_sentences.append(augment_sentence(sentence))

# Kombiniere Original- und augmentierte Sätze
all_sentences.extend(augmented_sentences)

# TF-IDF-Modell erstellen (statt Bag-of-Words)
vectorizer = TfidfVectorizer(max_features=2000)  # Maximal 2000 häufigste Features, um den Speicherverbrauch zu kontrollieren
X = vectorizer.fit_transform(all_sentences).toarray()

# Zielklassen (One-Hot-Encoding)
y = []
for doc in documents:
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    y.append(output_row)

y = np.array(y * 2)  # Verdopple die Anzahl der Labels, um sie an die augmentierten Daten anzupassen

# Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Class Weights basierend auf der Häufigkeit der Klassen berechnen
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
class_weight_dict = dict(enumerate(class_weights))

# Modell erstellen (z.B. einfaches Dense Neural Network)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(300, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),  # L2-Regularisierung
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

# Kompilieren mit Adam Optimizer
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.007), metrics=['accuracy'])

# Early Stopping hinzufügen, um Overfitting zu vermeiden
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

# Modell trainieren mit Class Weights
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, batch_size=16, class_weight=class_weight_dict, callbacks=[early_stopping,lr_scheduler])

# Modell und Vectorizer speichern
model.save('chatbot_model.h5')
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Modell mit TF-IDF, Augmentierung und Class Weights wurde erfolgreich trainiert und gespeichert.")