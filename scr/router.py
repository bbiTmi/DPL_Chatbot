import random
import pandas as pd
from intent.predict import predict_intent
from order.handle_order import handle_order_message
import os

response_df = pd.read_csv('response_data.csv')

def get_random_response(intent):
    matched = response_df[response_df['intent'] == intent]
    if matched.empty:
        return "Sorry I don't understand"
    return random.choice(matched['response'].values)

def route_user_input(text):
    intent, conf = predict_intent(text)

    print(f"Intent: {intent} (conf = {conf:.2f})")

    if intent in ("order", "change_order", "cancel_order"):
        return handle_order_message(text)
    else:
        return get_random_response(intent)
    
if __name__ == "__main__":
    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"): break
        reply = route_user_input(user)
        print("Bot:", reply)
