from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI


class Basket(BaseModel):
    item_name: str
    quantity: int


class Results(BaseModel):
    intent: str
    basket: list[Basket]


class LlmManager:

    INTENTS = [
        "ADD_ITEM",
        "REMOVE_ITEM",
        "UPDATE_QUANTITY",
        "CLEAR_BASKET",
        "EXIT",
        "INVALID_QUERY",
        "CHECKOUT",
        "RESET",
        "START"
    ]
    PRODUCTS = [
        "playstation 5",
        "nintendo switch",
        "pokemon plush",
        "ipad",
        "xbox",
        "iphone"
    ]
    SYNONYMS = {
        "playstation 5": ["ps5", "playstation", "ps"],
        "nintendo switch": ["nintendo", "switch"],
        "pokemon plush": ["plush", "pokemon"],
        "ipad": ["apple ipad", "apple tablet"],
        "xbox": ["microsoft xbox"],
        "iphone": ["apple iphone"],
    }

    def __init__(self):

        load_dotenv()

        # Initialize OpenAI client
        self.client = OpenAI()

        # Global variables
        self.basket = {}
        self.stop_listening = False

    def process_user_input(self, input_text, basket):
        prompt = f"""
## CONTEXT:
You are a shopping assistant chatbot. The user will give commands to add, remove, or update items in their basket. Keep track of the basket accordingly.

## ROLE:
You must process the user's query and update the basket based on the query. You must accomodate for the following "intents" that a user may have:
- Adding an item
- Removing an item
- Updating the quantity of an item
- Clearing the basket
- Checking the current basket
- Exiting the shopping session
- Handling invalid queries
- Asking to checkout/pay

## STEPS:
1. Process the users query
2. Update the current basket based on the query
3. Return the results in the specified valid JSON structure

## RULES:
- YOU MUST handle irrelevant queries gracefully and respond with a helpful message stating your purpose
- In the JSON response, for the "intent" value, you MUST return one of these values: {self.INTENTS}
- When identifying items, you MUST match them to one of these values: {self.PRODUCTS}, if relevant
- Do NOT match items to a product name if the user is not referring to a relevant product
- You must handle synonyms for the product names. Use this lookup: {self.SYNONYMS} to help with this
"""
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Here is the current basket: {basket}. Process this query and provide the outputs: \"{input_text}\""
            }
        ]

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0,
            n=1,
            presence_penalty=0,
            response_format=Results,
            seed=42
        )

        response = completion.choices[0].message
        if response.parsed:
            parsed = response.parsed
            return {
                "intent": parsed.intent,
                "basket": [{"item_name": item.item_name, "quantity": item.quantity} for item in parsed.basket],
            }
        elif response.refusal:
            print(f"REFUSED: {response.refusal}\n")
            return None
        return None
