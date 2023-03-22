import os
from decouple import config

import telebot
from joblib import load 
import pandas as pd
import sklearn


BOT_TOKEN = config('BOT_TOKEN')
print(BOT_TOKEN)

bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Howdy, how are you doing?")


@bot.message_handler(commands=['predict_flight'])
def sign_handler(message):
    text = "Please wright a message containing day of the week, planned departure and arrival times, carrier, origin and destination airports codes written with ', ' inbetween"
    sent_msg = bot.send_message(message.chat.id, text, parse_mode="Markdown")
    bot.register_next_step_handler(sent_msg, predict_handler)


def predict_handler(message):
    sign = message.text
    inputs = sign.split(', ')
    model = load('filename.joblib')
    scaler = load('scaler.joblib')
    cols = load('list_of_columns.joblib')
    s = pd.Series([0]*len(cols),index=cols)
    day, dep_time, arr_time, carrier, origin, dest = inputs
    s['DAY_OF_WEEK'], s['DEP_TIME'], s['ARR_TIME'], s[f"OP_CARRIER_{carrier}"] = day, dep_time, arr_time, 1
    s[f'ORIGIN_{origin}'], s[f'DEST_{dest}'] = 1, 1
    dfdf = pd.DataFrame(columns=cols)
    dfdf = pd.concat([dfdf, s.to_frame().T], ignore_index=True)
    X_cat = dfdf.iloc[:, 3:]
    X_num = dfdf.iloc[0,:3]

    X_scaled = scaler.transform(X_num.array.reshape(1, -1))
    X_scaled = pd.DataFrame(X_scaled, columns=X_num.index.values.tolist() )
    X = pd.concat([X_scaled, X_cat], axis=1)
    res = model.predict(X)
    
    bot.send_message(message.chat.id, res, parse_mode="Markdown")
    


@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    bot.reply_to(message, message.text)


bot.infinity_polling()