import telebot
import string
import pickle
import re
import csv
import pandas as pd
import numpy as np
import smtplib
import ssl
from email.message import EmailMessage
from translate import Translator
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from gtts import gTTS
import os
import phonenumbers
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests

# Replace with your Telegram bot API key
API_KEY = "6414383561:AAFXWZj8MW0f2GWpFehgUjMlLOLUYemqe50"
# Replace with your HERE API key
HERE_API_KEY = "8O8PZl_2YyxCpw7uaJCfXVbLopmyDG5ZJdphlo9UaZw"

bot = telebot.TeleBot(API_KEY)

# Default language code to translate to
DEFAULT_TRANSLATION_LANGUAGE = "en"

# Initialize translators
translator_to_english = Translator(to_lang=DEFAULT_TRANSLATION_LANGUAGE)

# Dictionary to store user preferences
user_preferences = {}

# Supported languages
supported_languages = {
    'af': 'Afrikaans (V)',
    'sq': 'Albanian (NV)',
    'ar': 'Arabic (V)',
    'hy': 'Armenian (NV)',
    'az': 'Azerbaijani (NV)',
    'eu': 'Basque (NV)',
    'be': 'Belarusian (NV)',
    'bn': 'Bengali (V)',
    'bs': 'Bosnian (V)',
    'bg': 'Bulgarian (NV)',
    'ca': 'Catalan (V)',
    'ceb': 'Cebuano (NV)',
    'zh-CN': 'Chinese (Simplified) (V)',
    'zh-TW': 'Chinese (Traditional) (V)',
    'hr': 'Croatian (V)',
    'cs': 'Czech (V)',
    'da': 'Danish (V)',
    'nl': 'Dutch (V)',
    'en': 'English (V)',
    'eo': 'Esperanto (NV)',
    'et': 'Estonian (NV)',
    'tl': 'Filipino (V)',
    'fi': 'Finnish (V)',
    'fr': 'French (V)',
    'gl': 'Galician (NV)',
    'ka': 'Georgian (NV)',
    'de': 'German (V)',
    'el': 'Greek (V)',
    'gu': 'Gujarati (V)',
    'ht': 'Haitian Creole (NV)',
    'he': 'Hebrew (NV)',
    'hi': 'Hindi (V)',
    'hu': 'Hungarian (V)',
    'is': 'Icelandic (V)',
    'id': 'Indonesian (V)',
    'ga': 'Irish (NV)',
    'it': 'Italian (V)',
    'ja': 'Japanese (V)',
    'jv': 'Javanese (V)',
    'kn': 'Kannada (V)',
    'kk': 'Kazakh (NV)',
    'km': 'Khmer (V)',
    'ko': 'Korean (V)',
    'ku': 'Kurdish (NV)',
    'ky': 'Kyrgyz (NV)',
    'lo': 'Lao (NV)',
    'la': 'Latin (V)',
    'lv': 'Latvian (V)',
    'lt': 'Lithuanian (V)',
    'lb': 'Luxembourgish (NV)',
    'mk': 'Macedonian (NV)',
    'mg': 'Malagasy (NV)',
    'ms': 'Malay (V)',
    'ml': 'Malayalam (V)',
    'mt': 'Maltese (NV)',
    'mi': 'Maori (NV)',
    'mr': 'Marathi (V)',
    'mn': 'Mongolian (NV)',
    'ne': 'Nepali (V)',
    'no': 'Norwegian (V)',
    'ps': 'Pashto (NV)',
    'fa': 'Persian (NV)',
    'pl': 'Polish (V)',
    'pt': 'Portuguese (V)',
    'pa': 'Punjabi (V)',
    'ro': 'Romanian (V)',
    'ru': 'Russian (V)',
    'sm': 'Samoan (NV)',
    'gd': 'Scots Gaelic (NV)',
    'sr': 'Serbian (V)',
    'si': 'Sinhala (V)',
    'sk': 'Slovak (V)',
    'sl': 'Slovenian (V)',
    'so': 'Somali (NV)',
    'es': 'Spanish (V)',
    'su': 'Sundanese (V)',
    'sw': 'Swahili (V)',
    'sv': 'Swedish (V)',
    'tg': 'Tajik (NV)',
    'ta': 'Tamil (V)',
    'te': 'Telugu (V)',
    'th': 'Thai (V)',
    'tr': 'Turkish (V)',
    'uk': 'Ukrainian (V)',
    'ur': 'Urdu (V)',
    'uz': 'Uzbek (NV)',
    'vi': 'Vietnamese (V)',
    'cy': 'Welsh (V)',
    'xh': 'Xhosa (NV)',
    'yi': 'Yiddish (NV)',
    'yo': 'Yoruba (NV)',
    'zu': 'Zulu (NV)'
}



# Core symptoms
core_symptoms = ["fever", "cough", "fatigue", "headache", "loss of smell or taste"]

core_symptom_ids = {
    "cough": "1",
    "fever": "2",
    "headache": "3",
    "nausea": "4",
    "sore throat": "5",
}

done_button_id = "6"

# Create a dictionary to store the chat log for each user
chat_logs = {}

training = pd.read_csv('Training.csv')
reduced_data = training.groupby(training['prognosis']).max()

le = None
le2 = None
model, model2 = None, None

# Load le from saved pickle after training
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

with open('label_encoder2.pkl', 'rb') as f:
    le2 = pickle.load(f)

with open('model1.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model2.pkl', 'rb') as f:
    model2 = pickle.load(f)

symptomsDict = {symptom: index for index, symptom in enumerate(model.feature_names_in_)}
description_list = dict()

# Initialize descriptions list
with open('/content/drive/MyDrive/Project/symptom_Description.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        description_list[row[0]] = row[1]

states = dict()  # Store current user text
store = dict()  # Store user answer

FINAL = 6

def check_pattern(dis_list, _input):
    _input = _input.replace(' ', '_')
    pattern = f"{_input}"
    regexp = re.compile(pattern)
    pred_list = [item for item in dis_list if regexp.search(item)]
    return pred_list

def predict_from_symptoms(symptoms, use_other_model=False):
    inputVector = np.zeros(len(model.feature_names_in_))  # Empty vector to be filled
    for s in symptoms:
        inputVector[symptomsDict[s]] = 1  # Fill symptoms vector
    if not use_other_model:
        symptom_pred = model.predict([inputVector])
        disease = le.inverse_transform(symptom_pred)
        return disease
    else:
        symptom_pred = model2.predict([inputVector])
        disease = le2.inverse_transform(symptom_pred)
        return disease

def preprocess(symptom, duration):
    symptom = symptom.lower().strip().translate(str.maketrans('', '', string.punctuation))
    duration_days = duration
    symptoms = ["fever", "cough", "fatigue", "headache", "loss of smell or taste"]
    symptom_one_hot = [int(symptom == s) for s in symptoms]
    data = [symptom_one_hot + [duration_days]]
    return data

def send_email_with_chat_log(chat_log, diagnosis):
    email_sender = 'joshua.healthchatbot@gmail.com'
    email_password = 'vgjphuxoxavaadri'
    email_receiver = 'jjsam106@gmail.com'
    subject = 'Health Chatbot Medical Diagnosis'
    body = f"From all symptoms provided, it seems this patient may have {diagnosis}. Please look to the chat dialog below and contact the user for further treatment. \n\nChat Log:\n{chat_log}"
    em = EmailMessage()
    em['From'] = email_sender
    em['To'] = email_receiver
    em['Subject'] = subject
    em.set_content(body)
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
        smtp.login(email_sender, email_password)
        smtp.sendmail(email_sender, email_receiver, em.as_string())

@bot.message_handler(commands=['start', 'help'])
def send_language_selection(message):
    _id = message.chat.id
    keyboard = ReplyKeyboardMarkup(resize_keyboard=True)
    for code, language in supported_languages.items():
        keyboard.row(KeyboardButton(text=language))
    bot.send_message(_id, "Hello, please select your preferred language:", reply_markup=keyboard)

@bot.message_handler(func=lambda message: message.text in supported_languages.values() and message.chat.id not in user_preferences)
def handle_language_selection(message):
    _id = message.chat.id
    selected_language = message.text
    user_preferences[_id] = {'language': selected_language, 'code': list(supported_languages.keys())[list(supported_languages.values()).index(selected_language)]}
    translator = Translator(to_lang=user_preferences[_id]['code'])
    name_prompt = translator.translate("What's your name?")
    bot.send_message(_id, name_prompt)
    send_voice_message(_id, name_prompt)
    bot.register_next_step_handler(message, start_repl)

def start_repl(message):
    _id = message.chat.id
    store[_id] = {'symptoms': []}  # Initialize store with symptoms key
    store[_id]["name"] = message.text
    chat_logs[_id] = []
    chat_logs[_id].append(f"User Name: {message.text}")
    bot.register_next_step_handler(message, get_phone_number)
    send_translated_message(_id, f"Hello, {store[_id]['name']}. What's your phone number?")

def get_phone_number(message):
    _id = message.chat.id
    phone_number = message.text
    if validate_phone_number(phone_number):
        if user_preferences[_id]['code'] == "en":            
          store[_id]["phone_number"] = phone_number
          chat_logs[_id].append(f"User Phone No.: {message.text}")
          bot.register_next_step_handler(message, get_symptoms)
          send_translated_message(_id, "Great! What symptoms are you experiencing?")
          send_core_symptom_buttons(_id)
        else:
          store[_id]["phone_number"] = phone_number
          chat_logs[_id].append(f"User Phone No.: {message.text}")
          bot.register_next_step_handler(message, get_symptoms)
          send_translated_message(_id, "Great! What symptoms are you experiencing? (Reply with numbers)")
          send_core_symptom_buttons(_id)
    else:
        bot.register_next_step_handler(message, get_phone_number)
        send_translated_message(_id, "Invalid phone number. Please enter a valid phone number with country code.")

def validate_phone_number(phone_number):
    try:
        parsed_number = phonenumbers.parse(phone_number)
        return phonenumbers.is_valid_number(parsed_number)
    except phonenumbers.NumberParseException:
        return False

def get_symptoms(message):
      _id = message.chat.id
      user_response = message.text
      
      if user_preferences[_id]['code'] != "en": 
        _id = message.chat.id
        user_response = message.text
        
        # Check if the response matches any symptom IDs
        if user_response in core_symptom_ids.values():
            symptom_key = list(core_symptom_ids.keys())[list(core_symptom_ids.values()).index(user_response)]
            store[_id]['symptoms'].append(symptom_key)
            bot.register_next_step_handler(message, get_symptoms)
            send_translated_message(_id, "Thank you for sharing your symptom. If you have any more to share, please go ahead, else click 'Done'.")
            send_core_symptom_buttons(_id)
            return
        
        # Check if the response matches the 'done' ID
        if user_response == done_button_id:
            bot.register_next_step_handler(message, get_symptom_days)
            send_translated_message(_id, "How many days has this occurred?")
            return
        
        latest_response = translate_to_english(_id, user_response)
        latest_response = latest_response.replace(" ", "_")
        chat_logs[_id].append(f"User symptom: {latest_response}")
        
        if latest_response in symptomsDict.keys():
            store[_id]['symptoms'].append(latest_response)
            bot.register_next_step_handler(message, get_symptoms)
            send_translated_message(_id, "Thank you for sharing your symptom. If you have any more to share, please go ahead, else click 'Done'.")
            send_core_symptom_buttons(_id)
            return
        
        pred_list = check_pattern(symptomsDict.keys(), latest_response)
        if pred_list:
            resp = "Did you mean any of the following? Reply with the updated name"
            for idx, i in enumerate(pred_list):
                resp += f"\n {idx}) {i}"
            bot.register_next_step_handler(message, get_symptoms)
            resp = resp.replace("_"," ")
            send_translated_message(_id, resp)
            return
        else:
            bot.register_next_step_handler(message, get_symptoms)
            send_translated_message(_id, "Sorry this symptom is not in our database. Try again.")
            return
      else:
        _id = message.chat.id
        latest_response = translate_to_english(_id, message.text)
        latest_response = latest_response.replace(" ", "_")
        chat_logs[_id].append(f"User symptom: {latest_response}")
        
        if latest_response.lower() == translate_to_english(_id, "done").lower():
            bot.register_next_step_handler(message, get_symptom_days)
            send_translated_message(_id, "How many days has this occurred?")
            return
        
        if latest_response in symptomsDict.keys():
            store[_id]['symptoms'].append(latest_response)
            bot.register_next_step_handler(message, get_symptoms)
            send_translated_message(_id, "Thank you for sharing your symptom. If you have any more to share, please go ahead, else click 'Done'.")
            send_core_symptom_buttons(_id)
            return
        
        pred_list = check_pattern(symptomsDict.keys(), latest_response)
        if pred_list:
            resp = "Did you mean any of the following? Reply with the updated name"
            for idx, i in enumerate(pred_list):
                resp += f"\n {idx}) {i}"
            bot.register_next_step_handler(message, get_symptoms)
            resp = resp.replace("_"," ")
            send_translated_message(_id, resp)
            return
        else:
            bot.register_next_step_handler(message, get_symptoms)
            send_translated_message(_id, "Sorry this symptom is not in our database. Try again.")
            return

def get_symptom_days(message):
    _id = message.chat.id
    if message.text.isnumeric():
        store[_id]["duration"] = int(message.text)
        chat_logs[_id].append(f"Symptom Duration (days): {message.text}")
        bot.register_next_step_handler(message, get_other_symptoms)
        Okay_button = KeyboardButton(translate_to_user_language(_id, "okay"))
        keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        keyboard.add(Okay_button)
        bot.send_message(_id,translate_to_user_language(_id, "I'm going to ask you some follow-up questions. Just respond with yes/no. Click 'okay' to proceed ."), reply_markup=keyboard)
        send_voice_message(_id, translate_to_user_language(_id, "I'm going to ask you some follow-up questions. Just respond with yes/no. Click 'okay' to proceed ."))
    else:
        bot.register_next_step_handler(message, get_symptom_days)
        send_translated_message(_id, "Response was not a number. Try again.")


def get_other_symptoms(message):
    _id = message.chat.id
    if _id not in states:  # Initialize states for the chat if it doesn't exist
        states[_id] = 0
    if "symptoms_given" not in store[_id]:
        symptoms = store[_id]["symptoms"]
        present_disease = predict_from_symptoms(symptoms)
        red_cols = reduced_data.columns
        symptoms_given = symptoms
        symptoms_not_given = [sym for sym in red_cols if sym not in symptoms_given]
        store[_id]["symptoms_not_given"] = symptoms_not_given
        store[_id]["disease"] = present_disease
        store[_id]["symptoms_given"] = symptoms_given
    else:
        curr_state = states[_id]
        response = translate_to_english(_id, message.text).lower()
        if response == "yes":
            symptoms_given = store[_id]["symptoms_given"]
            other_symptom = store[_id]["symptoms_not_given"][curr_state]
            symptoms_given.append(other_symptom)
            present_disease = predict_from_symptoms(symptoms_given)
            store[_id]["disease"] = present_disease
            store[_id]["symptoms_given"] = symptoms_given
        states[_id] += 1
    curr_state = states[_id]
    if curr_state >= FINAL:
        present_disease = store[_id]["disease"]
        name = store[_id]["name"]
        chat_log = "\n".join(chat_logs[_id])
        send_email_with_chat_log(chat_log, present_disease[0])
        send_translated_message(_id, f"Thank you, {name}. I predict you may have {present_disease[0]}. {description_list[present_disease[0]]}.")
        send_nearby_hospital_button(_id)
    else:
        other_symptom = store[_id]["symptoms_not_given"][curr_state]
        other_symptom = other_symptom.replace("_", " ")
        bot.register_next_step_handler(message, get_other_symptoms)
        send_followup_buttons(_id, f"Are you experiencing {other_symptom}?")


def translate_to_english(chat_id, message):
    if chat_id in user_preferences:
        language_code = user_preferences[chat_id]['code']
    else:
        language_code = DEFAULT_TRANSLATION_LANGUAGE
    translator = Translator(from_lang=language_code, to_lang="en")
    translated_message = translator.translate(message)
    return translated_message


def translate_to_user_language(chat_id, message):
    if chat_id in user_preferences:
        language_code = user_preferences[chat_id]['code']
    else:
        language_code = DEFAULT_TRANSLATION_LANGUAGE
    translator = Translator(to_lang=language_code)
    translated_message = translator.translate(message)
    return translated_message



def send_voice_message(chat_id, message):
    if '(V)' in user_preferences[chat_id]['language']:
        tts = gTTS(text=message, lang=user_preferences[chat_id]['code'])
        tts.save('voice_message.mp3')
        with open('voice_message.mp3', 'rb') as voice:
            bot.send_voice(chat_id, voice)
        os.remove('voice_message.mp3')
    else:
        return  # Do nothing and let the bot continue as normal



def send_core_symptom_buttons(chat_id):
    if user_preferences[chat_id]['code'] == "en":
        keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        buttons = [KeyboardButton(translate_to_user_language(chat_id, symptom)) for symptom in core_symptoms]
        keyboard.add(*buttons)
        keyboard.add(KeyboardButton(translate_to_user_language(chat_id, "Done")))
        bot.send_message(chat_id, translate_to_user_language(chat_id, "Please select your core symptoms or type them manually:"), reply_markup=keyboard)
        send_voice_message(chat_id, translate_to_user_language(chat_id, "Please select your core symptoms or type them manually:"))

    else:  
        keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
        buttons = [KeyboardButton(f"{translate_to_user_language(chat_id, symptom)} ({core_symptom_ids[symptom]})") for symptom in core_symptom_ids]
        keyboard.add(*buttons)
        keyboard.add(KeyboardButton(f"{translate_to_user_language(chat_id, 'Done')} ({done_button_id})"))
        bot.send_message(chat_id, translate_to_user_language(chat_id, "Please select your core symptoms or type them manually:"), reply_markup=keyboard)
        send_voice_message(chat_id, translate_to_user_language(chat_id, "Please select your core symptoms or type them manually:"))


@bot.message_handler(func=lambda message: True)
def handle_symptom_selection(message):
    chat_id = message.chat.id
    user_input = message.text
    
    # Check if the input contains an ID
    match = re.search(r'\((\d+)\)', user_input)
    if match:
        symptom_id = int(match.group(1))
        
        if symptom_id == done_button_id:
            # Handle the 'Done' action
            bot.send_message(chat_id, translate_to_user_language(chat_id, "Thank you! You've completed the symptom selection."))
        else:
            # Process the selected symptom by ID
            selected_symptom = get_symptom_by_id(symptom_id)
            bot.send_message(chat_id, translate_to_user_language(chat_id, f"You've selected: {selected_symptom}"))
            
            # Here you can add further processing for the selected symptom
    else:
        # Handle manual symptom entry
        bot.send_message(chat_id, translate_to_user_language(chat_id, "Please enter a valid symptom or select from the list."))

def get_symptom_by_id(symptom_id):
    for symptom, id_ in core_symptom_ids.items():
        if id_ == symptom_id:
            return symptom
    return None



def send_followup_buttons(chat_id, question):
    translated_question = translate_to_user_language(chat_id, question)
    yes_button = KeyboardButton(translate_to_user_language(chat_id, "Yes"))
    no_button = KeyboardButton(translate_to_user_language(chat_id, "No"))
    keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    keyboard.add(yes_button, no_button)
    bot.send_message(chat_id, translated_question, reply_markup=keyboard)
    send_voice_message(chat_id, translated_question)

def send_nearby_hospital_button(chat_id):
    keyboard = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    hospital_button = KeyboardButton(translate_to_user_language(chat_id, "🏥"))
    keyboard.add(hospital_button)
    bot.send_message(chat_id,translate_to_user_language(chat_id, "Please click the button to find nearby hospitals."), reply_markup=keyboard)
    send_voice_message(chat_id, translate_to_user_language(chat_id, "Please click the button to find nearby hospitals."))

@bot.message_handler(func=lambda message: translate_to_english(message.chat.id, message.text).lower() == "🏥")
def handle_find_nearby_hospital(message):
    chat_id = message.chat.id
    send_location_buttons(chat_id)


share_location_button_id = "share_location"
type_location_button_id = "type_location"

def send_location_buttons(chat_id):
  if user_preferences[chat_id]['code'] != "en":
    keyboard = ReplyKeyboardMarkup(row_width=2)
    share_location_button = KeyboardButton(f"{translate_to_user_language(chat_id, 'Share location (Mobile App)')} ({share_location_button_id})", request_location=True)
    type_location_button = KeyboardButton(f"{translate_to_user_language(chat_id, 'Type Location (Telegram Desktop)')} ({type_location_button_id})")
    keyboard.add(share_location_button, type_location_button)
    bot.send_message(chat_id, translate_to_user_language(chat_id, "Please click the button to find nearby hospitals."), reply_markup=keyboard)
    send_voice_message(chat_id, translate_to_user_language(chat_id, "Please click the button to find nearby hospitals."))
  else:
    keyboard = ReplyKeyboardMarkup(row_width=2)
    share_location_button = KeyboardButton("Share location (Mobile App)", request_location=True)
    type_location_button = KeyboardButton("Type Location (Telegram Desktop)")
    keyboard.add(share_location_button, type_location_button)
    bot.send_message(chat_id,translate_to_user_language(chat_id, "Please click the button to find nearby hospitals."), reply_markup=keyboard)
    send_voice_message(chat_id, translate_to_user_language(chat_id, "Please click the button to find nearby hospitals."))


@bot.message_handler(content_types=['location'])
def handle_shared_location(message):
    chat_id = message.chat.id
    location = message.location
    if location:
        latitude = location.latitude
        longitude = location.longitude
        suggest_nearby_hospitals(chat_id, latitude, longitude)

@bot.message_handler(func=lambda message: message.text.lower() == 'type location (telegram desktop)')
def handle_manual_location_request(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "Please type your location address (Include State and Country):")
    bot.register_next_step_handler(message, handle_manual_location)

def handle_manual_location(message):
    chat_id = message.chat.id
    location_address = message.text
    geocode_location(chat_id, location_address)

def geocode_location(chat_id, address):
    params = {
        'q': address,
        'apiKey': HERE_API_KEY
    }
    response = requests.get("https://geocode.search.hereapi.com/v1/geocode", params=params)
    data = response.json()
    if data['items']:
        location = data['items'][0]['position']
        latitude = location['lat']
        longitude = location['lng']
        suggest_nearby_hospitals(chat_id, latitude, longitude)
    else:
        bot.send_message(chat_id, "Sorry, I couldn't find that location. Please try again.")

def suggest_nearby_hospitals(chat_id, latitude, longitude):
    params = {
        'apiKey': HERE_API_KEY,
        'at': f"{latitude},{longitude}",
        'q': 'hospital',
        'limit': 5
    }

    response = requests.get("https://discover.search.hereapi.com/v1/discover", params=params)
    places = response.json().get('items', [])

    if places:
        message = "Here are some nearby hospitals:"
        for place in places:
            name = place.get('title', 'No name available')
            address = place.get('address', {}).get('label', 'No address available')
            place_lat = place['position']['lat']
            place_lng = place['position']['lng']
            # Google Maps link for directions
            maps_link = f"https://www.google.com/maps/dir/?api=1&origin={latitude},{longitude}&destination={place_lat},{place_lng}"
            message += f"\n- {name}, {address} [Directions]({maps_link})"
    else:
        message = "Sorry, no hospitals found nearby."

    bot.send_message(chat_id, message, parse_mode='Markdown')


def send_translated_message(chat_id, message):
    language_code = user_preferences[chat_id]['code']
    translator = Translator(to_lang=language_code)
    translated_message = translator.translate(message)
    bot.send_message(chat_id, translated_message)
    send_voice_message(chat_id, translated_message)




bot.polling()
