import sqlite3
from word2number import w2n
from .slot_predict import predict_slots
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "coffee.db")
DB_PATH = os.path.abspath(DB_PATH)
global pending_orders
pending_orders = []

def extract_entities(token_slot_pairs):
    orders = []
    current_item = []
    current_qty = None

    for token, tag in token_slot_pairs:
        if tag == "B-quantity":
            try:
                current_qty = int(token) if token.isdigit() else w2n.word_to_num(token)
            except:
                current_qty = 1

        elif tag in ("B-item", "I-item"):
            current_item.append(token)

        elif tag == "O" and current_item:
            item_name = " ".join(current_item)
            orders.append({"item": item_name, "qty": current_qty or 1})
            current_item = []
            current_qty = None

    if current_item:
        item_name = " ".join(current_item)
        orders.append({"item": item_name, "qty": current_qty or 1})

    return orders

def is_item_in_menu(item_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT item FROM menu")
    menu_items = [row[0].lower() for row in cursor.fetchall()]
    conn.close()
    return item_name.lower() in menu_items

def save_order(item, qty):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO orders (item, qty) VALUES (?, ?)", (item, qty))
    conn.commit()
    conn.close()

def handle_order_message(message):
    token_slot_pairs = predict_slots(message)
    item, qty = extract_entities(token_slot_pairs)

    if not item:
        return "Sorry! Please try again"

    if not is_item_in_menu(item):
        return f"{item} not in our menu. Please choose another drinks"

    save_order(item, qty)
    return f"Order completed! Your order: {qty}, {item}"

def format_orders():
    if not pending_orders:
        return "Not any item in order list"
    return "\n".join([f"- {o['qty']} {o['item']}" for o in pending_orders])

def handle_order_message(message):
    global pending_orders
    msg = message.lower().strip()

    if msg in ("done", "complete"):
        if not pending_orders:
            return "You haven't ordered yet! Please choose a drink"
        for order in pending_orders:
            save_order(order["item"], order["qty"])
        response = f"Order completed! Your order: \n{format_orders()}"
        pending_orders.clear()
        return response

    if "cancel" in msg:
        token_slot_pairs = predict_slots(message)
        item_orders = extract_entities(token_slot_pairs)
        if not item_orders:
            return "Which drink do you want to cancel?"
        for item in item_orders:
            pending_orders = [o for o in pending_orders if o["item"].lower() != item["item"].lower()]
        return f"Canceled. Your order: \n{format_orders()}"

    if "change" in msg:
        token_slot_pairs = predict_slots(message)
        item_orders = extract_entities(token_slot_pairs)
        if not item_orders:
            return "Which drink do you want to change?"
        updated = []
        for item in item_orders:
            for o in pending_orders:
                if o["item"].lower() == item["item"].lower():
                    o["qty"] = item["qty"]
                    updated.append(item["item"])
        if not updated:
            return "You haven't ordered yet! Not found"
        return f"Updated. Your order: \n{format_orders()}"

    # Mặc định: thêm món mới
    token_slot_pairs = predict_slots(message)
    item_orders = extract_entities(token_slot_pairs)
    if not item_orders:
        return "Sorry I don't understand"

    added = []
    for order in item_orders:
        if is_item_in_menu(order["item"]):
            pending_orders.append(order)
            added.append(f"- {order['qty']} {order['item']}")

    if not added:
        return "Not found in our menu. Please choose another drink"

    return f"Added:\n{chr(10).join(added)}\n👉 Your order:\n{format_orders()}\n Do you want to change/cancel or complete your order"
