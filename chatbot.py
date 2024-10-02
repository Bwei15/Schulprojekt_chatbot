#
#   Version 1.0
#

import json
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification

error_message = [
    "Entschuldigung, ich habe das nicht ganz verstanden. Könntest du das bitte umformulieren?", 
    "Ich bin mir nicht sicher, ob ich das richtig verstehe. Kannst du das genauer erklären?", 
    "Hm, das scheint nicht korrekt zu sein. Kannst du es anders ausdrücken?", 
    "Ich habe Schwierigkeiten, deine Anfrage zu verstehen. Kannst du mehr Details geben?", 
    "Das habe ich leider nicht verstanden. Kannst du es anders sagen?", 
    "Entschuldigung, aber ich bin mir nicht sicher, was du meinst. Kannst du es noch einmal versuchen?", 
    "Tut mir leid, aber das verwirrt mich. Kannst du deine Frage präzisieren?", 
    "Ich weiß nicht, wie ich darauf antworten soll. Kannst du es anders formulieren?", 
    "Es scheint, dass ich deine Anfrage nicht verstehe. Kannst du es anders sagen?", 
    "Das habe ich nicht ganz mitbekommen. Könntest du es auf eine andere Weise sagen?"
]
error = open("error.txt","a")

tokenizer = BertTokenizer.from_pretrained('bert_intent_model')
model = BertForSequenceClassification.from_pretrained('bert_intent_model')

with open('label_mapping.json', 'r') as file:
    label_mapping = json.load(file)
    label_mapping = {int(k): v for k, v in label_mapping.items()}

with open('train_data.json', 'r') as file:
    intents_data = json.load(file)

def encode_text(text):
    return tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors='pt', return_attention_mask=True)

def predict_intent(text, threshold=0.55):
    encoding = encode_text(text)
    with torch.no_grad():
        outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'])
        probabilities = torch.softmax(outputs.logits, dim=1) 
        predicted_probability, predicted_label = torch.max(probabilities, dim=1)
        predicted_probability = predicted_probability.item()
        if predicted_probability >= threshold:
            return label_mapping[predicted_label.item()], predicted_probability
        else:
            return None, predicted_probability

def get_response(intent):
    for intent_data in intents_data['intents']:
        if intent_data['tag'] == intent:
            return random.choice(intent_data['responses'])
    return "Entschuldigung, es ist ein Fehler Aufgetreten."

def get_response_error():
    return random.choice(error_message)

def main_chat():
    print("Starten...\n")
    print("Botversion 1.0\n\n")
    print("Info: Durch die Nutzung dieses Chatbots stimmen Sie automatisch den Nutzungsbedingungen zu: \nAlle von Ihnen eingegebenen Daten und Nachrichten werden gespeichert und können zur \nVerbesserung des Bots verwendet werden und ausgewertet werden. \nZum Verlassen des Bots schreiben sie exit.\n\n")
    print("Chatbot: Hey, was kann  ich für Sie tun?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Bye...")
            break
        predicted_intent, probability = predict_intent(user_input, threshold=0.55)
        if predicted_intent:
            response = get_response(predicted_intent)
            print(f"Chatbot: {response}  ({predicted_intent}, {probability:.2f})")
        else:
            error.write(f"\r\nError NICHT ERKANNT  -  (Wahrscheinlichkeit: {predicted_intent}  {probability:.2f})  |  USER INPUT:  {user_input} ")
            response = get_response_error()
            print(f"Chatbot: {response}")
main_chat()