import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pickle

# JSON-Daten laden
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()

# Das trainierte Modell, den TF-IDF Vectorizer und die Klassen laden
model = load_model('chatbot_model.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)

# Letze Antowrt
last_response = ""

# Globaler Schwellenwert für Vorhersagen
ERROR_THRESHOLD = 0.92

# Funktion, um eine Eingabe vom Benutzer in das TF-IDF-Format zu konvertieren
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ['?', '!', '.', ',']]
    return sentence_words

def tfidf_transform(sentence):
    # Konvertiere den Satz in die TF-IDF-Darstellung
    sentence_words = clean_up_sentence(sentence)
    sentence_tfidf = vectorizer.transform([' '.join(sentence_words)]).toarray()
    return sentence_tfidf

# Vorhersagefunktion
def predict_class(sentence):
    # Vorhersagen für die Klasse (Intent)
    tfidf_vector = tfidf_transform(sentence)
    res = model.predict(tfidf_vector)[0]

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    # Rückgabe von Intents mit ausreichender Wahrscheinlichkeit
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results] if results else []

# Antwort basierend auf der Vorhersage
def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return np.random.choice(i['responses'])  # Wähle eine zufällige Antwort
    return "Entschuldigung, ich verstehe nicht, was du meinst."

# Chatbot-Antwort
def chatbot_response(text):
    try:
        intents_list = predict_class(text)
        if not intents_list:
            return "Entschuldigung, ich verstehe nicht, was du meinst."
        response = get_response(intents_list, intents)
        return response
    except Exception as e:
        return f"Ein Fehler ist aufgetreten: {str(e)}"




# Developer-Bereich
def developer_mode():
    global ERROR_THRESHOLD
    print("\nDeveloper-Modus aktiviert! Du kannst jetzt zusätzliche Tests ausführen.")
    while True:
        dev_input = input("Developer: ").lower()
        if dev_input == "exit":
            print("Verlasse den Developer-Modus.")
            break
        elif dev_input == "show intents":
            print("\nKlassen (Intents):")
            print(classes)
        elif dev_input == "test":
            while True:
                sentence = input("Gib einen Satz ein, um die Vorhersage zu testen (oder 'exit' zum Beenden): ")
                if sentence.lower() == 'exit':
                    print("Vorhersagetests beendet.")
                    break
                prediction = predict_class(sentence)
                print(f"Vorhersage: {prediction}")
        elif dev_input == "set threshold":
            try:
                new_threshold = float(input("Setze neuen Schwellenwert (z. B. 0.25): "))
                if 0 < new_threshold < 1:
                    ERROR_THRESHOLD = new_threshold
                    print(f"Neuer Schwellenwert ist {ERROR_THRESHOLD}")
                else:
                    print("Der Schwellenwert muss zwischen 0 und 1 liegen.")
            except ValueError:
                print("Bitte eine gültige Zahl eingeben.")
        else:
            print("Unbekannter Befehl. Verfügbare Befehle: 'show intents', 'test prediction', 'set threshold', 'exit'.")

def tester_mode():
        print("\nHallo Tester! Danke, dass du dir Zeit nimmst um unseren Chatbot zu testen.")
        while True:
            tester_input = input("Tester: ").lower()
            if tester_input == "exit":
                print("Danke für den Testen. Damit hast du sehr geholfen!")
                break
            elif tester_input == "f":
                fehlermeldung = input("Dev: Was war Falsch? ")
                print("Den Fehler bitte abspeichern!")
                print("------------")
                print(f"{last_response} -------- {fehlermeldung}")
                print("------------")
            else:
                prediction = predict_class(tester_input)                
                response = chatbot_response(tester_input)
                print(f"Chatbot {prediction} : {response}")
                last_response = prediction

# Haupt-Chat-Funktion
def main_chat():
    print("***********************")
    print("Starten...")
    print("***********************")
    print("Chatbot: Was kann ich für dich tun?")

    while True:
        message = input("You: ").lower()
        if message == 'exit':
            print("Chatbot: Auf Wiedersehen!")
            break
        elif message == 'developer':
            developer_mode()
        elif message == 'tester':
            tester_mode()
        else:
            response = chatbot_response(message)
            print(f"Chatbot: {response}")

# Start des Chats
main_chat()