# create-your-own-ai-voice-agent

- Created at: 2025-03-04
- Created by: `🐢 Arun Godwin Patel @ Code Creations`

## Table of contents

- [Setup](#setup)
  - [System](#system)
  - [Installation](#installation)
- [Walkthrough](#walkthrough)
  - [Code Structure](#code-structure)
  - [Tech stack](#tech-stack)
  - [Build from scratch](#build-from-scratch)
    - [1. Create a virtual environment](#1-create-a-virtual-environment)
    - [2. Activate the virtual environment](#2-activate-the-virtual-environment)
    - [3. Install the required packages](#3-install-the-required-packages)
    - [4. Create the AI class](#4-create-the-ai-class)
    - [5. Create the Conversation class](#5-create-the-conversation-class)
    - [6. Create the main application](#6-create-the-main-application)

## Setup

### System

This code repository was tested on the following computers:

- Windows 10

At the time of creation, this code was built using `Python 3.9.13`

### Installation

1. Install `virtualenv`

```bash
# 1. Open a CMD terminal
# 2. Install virtualenv globally
pip install virtualenv
```

2. Create a virtual environment

```bash
python -m venv venv
```

3. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Mac
source venv/bin/activate
```

4. Install the required packages

```bash
pip install -r requirements.txt
```

5. Run the module

```bash
python main.py
```

## Walkthrough

### Code Structure

The code directory structure is as follows:

```plaintext
create-your-own-ai-voice-agent
└───classes
|   └──ai.py
|   └──conversation.py
│   .env
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
```

The `main.py` file is the entry point of the application. It imports the `ai` and `conversation` classes from the `classes` directory.

The `ai.py` file contains the `AI` class, which is responsible for handling the AI voice agent's functionality.

The `conversation.py` file contains the `Conversation` class, which is responsible for managing the conversation flow between the user and the AI voice agent.

The `.env` file contains the environment variables used by the application.

The `.gitignore` file specifies the files and directories that should be ignored by Git.

The `requirements.txt` file lists the Python packages required by the application.

### Tech stack

**AI**

- LLM: `OpenAI API`

**Voice**

- Text to Speech: `pyttsx3`
- Speech to Text: `SpeechRecognition` & `Google Web Speech API`

**Others**

- Asynchronous processing: `threading`
- Typing: `pydantic`
- Environment variables: `python-dotenv`

### Build from scratch

This project was built from scratch using Python and the OpenAI API. The AI voice agent can understand and respond to user queries related to shopping. The user can add, remove, update, or clear items in their basket, check the current basket, exit the shopping session, or checkout.

This code was tested using native English, but could be adapted to other languages by changing the AI & Voice models.

#### 1. Create a virtual environment

```bash
python -m venv venv
```

#### 2. Activate the virtual environment

```bash
# Windows
.\venv\Scripts\activate
# Mac
source venv/bin/activate
```

#### 3. Install the required packages

```bash
pip install -r requirements.txt
```

#### 4. Create the AI class

For this voice agent, we are using the `OpenAI` API to generate responses. You will need to create an account on the OpenAI website and save your API key in the `.env` file at the root of this repository.

Create a file called `ai.py` in the `classes` directory and add the following code:

```python
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI


class Basket(BaseModel):
    item_name: str
    quantity: int


class Results(BaseModel):
    intent: str
    basket: list[Basket]
```

We are importing the necessary packages and strict typing the data that we are expecting from the OpenAI API.

Next, let's start building our `LlmManager` class. This class will be responsible for interacting with the OpenAI API.

```python
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
```

In the `LlmManager` class, we define the possible intents that the user can have, the products that the user can interact with, and the synonyms for each product.

We also initialize the OpenAI client and set up some global variables to keep track of the user's basket.

Finally, we create the `process_user_input` method to handle incoming speech from the user and generate a response.

```python
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
```

This completes our `LlmManager` class. We have defined the possible intents, products, and synonyms that the user can interact with. We have also set up the OpenAI client and implemented the `process_user_input` method to handle incoming user queries.

#### 5. Create the Conversation class

Next, we will create the `Conversation` class, which will manage the conversation flow between the user and the AI voice agent.

Create a file called `conversation.py` in the `classes` directory and add the following code:

```python
import json
import time
import pyttsx3
import threading

import speech_recognition as sr
```

We are importing the necessary packages to handle speech recognition and text-to-speech conversion.

Next, let's start building our `Conversation` class.

```python
class ConversationManager:

    def __init__(self, queue, llm_manager):
        self.stop_listening = False
        self.energy_threshold = 5000
        self.speech_queue = queue
        self.llm_manager = llm_manager
```

We initialise the `ConversationManager` class with the necessary parameters and set up some global variables:

- `stop_listening`: A flag to stop the speech recognition loop
- `energy_threshold`: The energy threshold for speech recognition, this is used for noise cancelling and adjustment of the microphone sensitivity
- `speech_queue`: The queue to communicate with the AI thread
- `llm_manager`: The LlmManager instance to process user queries

Next, let's set up some helper methods to support the conversation flow.

```python
def _continuous_listen(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        recognizer.energy_threshold = self.energy_threshold

        def listen_loop():
            print("\n🎧 Preparing mic for continuous listening...")

            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=5)
                print("🎤 Calibrating microphone for ambient noise...")
                print(f"\t--> Initial energy threshold: {recognizer.energy_threshold}\n")

                while not self.stop_listening:
                    print("🎤...")
                    try:
                        audio = recognizer.listen(source, timeout=60, phrase_time_limit=90)
                        transcript = recognizer.recognize_google(audio)
                        print(f"\t--> 🗣️ You said: \"{transcript}\"")
                        self.speech_queue.put(transcript)  # Add the transcript to the queue
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        continue
                    except Exception:
                        continue

        # Start the listening loop in a background thread
        threading.Thread(target=listen_loop, daemon=True).start()
```

In the `_continuous_listen` method, we set up the speech recognition loop. We calibrate the microphone for ambient noise and continuously listen for user input. When the user speaks, we transcribe the audio and add it to the speech queue to be processed by the LLM.

Python is a blocking language, so we use threading to run the `_listen_loop` method in the background, so that the main thread can continue executing other tasks.

Let's continue.

```python
def _process_speech(self):
        while not self.stop_listening:
            try:
                # Wait for a speech event from the queue
                transcript = self.speech_queue.get(timeout=1)
                if not transcript.strip():
                    print("\t~~~ ❌ No valid input detected. Please try again.")
                    self._speak_response("I didn't catch that. Could you repeat your order?")
                    continue

                response = self.llm_manager.process_user_input(transcript, self.llm_manager.basket)
                if response is None:
                    print("\t~~~ ❌ Invalid query. Please try again.")
                    self.speak_response("I couldn't understand that. Please try again.")
                    continue

                intent = response.get("intent")

                if intent == "INVALID_QUERY":
                    print("\t❌ Invalid query. Please try again.")
                    self._speak_response("I didn't understand that. Please try again.")
                    continue

                if intent == "START":
                    print("\t🔄 Starting a new shopping session.")
                    self._speak_response("Welcome to KFC, what would you like to order")
                    self._save_state(response)
                    continue

                if intent == "RESET":
                    print("\t🔄 Resetting the shopping session.")
                    self._speak_response("Let's start again. How can I help you?")
                    basket = []
                    response["basket"] = basket
                    self._save_state(response)
                    continue

                if intent == "CLEAR_BASKET":
                    print("\t🧺 Clearing the basket.")
                    basket = []
                    response["basket"] = basket
                    self._speak_response("Basket cleared. Can I help with anything else?")
                    self._save_state(response)
                    continue

                if intent == "EXIT":
                    print("🔄 Exiting the application.")
                    self._save_state(response)
                    self._speak_response("Goodbye! Thank you for using the shopping assistant.")
                    self.stop_listening = True
                    break

                if intent == "CHECKOUT":
                    print(f"\t🧺 Checking out the basket:\n\n{json.dumps(self.llm_manager.basket, indent=4)}\n")
                    self._speak_response("Checking out your basket. Thank you for shopping with us.")
                    continue

                print(f"\t=== 🤖 Intent: {intent}")
                self.llm_manager.basket = response.get("basket", self.llm_manager.basket)
                print(f"\t=== 🤖 Updated basket:\n\n{json.dumps(self.llm_manager.basket, indent=4)}\n")
                self._speak_response("Can I help with anything else?")
                self._save_state(response)
            except Exception as ex:
                if not self.speech_queue.empty():
                    print(f"\t~~~⚠️ Error while processing speech: {ex}")
```

In the `_process_speech` method, we continuously process the user's speech input. We check the intent of the user's query and respond accordingly. We handle various intents such as starting a new shopping session, resetting the session, clearing the basket, exiting the application, checking out the basket, and updating the basket based on the user's query.

We also save the state of the conversation after each interaction to maintain the context of the conversation.

Finally, we create the `_speak_response` and `_save_state` methods to handle text-to-speech conversion and state saving.

```python
def _speak_response(self, response):
    def speech_thread():
        # try:
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()
        # except Exception as e:
        #     print(f"❌ Error playing speech: {e}")

    # Start the audio playback in a separate thread
    threading.Thread(target=speech_thread, daemon=True).start()

# Function to save the current application state
def _save_state(self, state):
    with open("response.json", "w") as f:
        json.dump(state, f, indent=4)
```

In the `_speak_response` method, we use the `pyttsx3` library to convert text to speech to play the audio response to the user.

In the `_save_state` method, we save the current application state to a JSON file to maintain the context of the conversation.

These helper methods are used by the `start` method, which is our entry point to the conversation flow.

```python
def start(self):

    # Start continuous listening
    self._continuous_listen()

    # Start processing speech events in a separate thread
    processor_thread = threading.Thread(target=self._process_speech, daemon=True)
    processor_thread.start()

    try:
        while not self.stop_listening:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("\n🔄 Shutting down...")
        self.stop_listening = True  # Signal threads to stop
        processor_thread.join()  # Wait for processor thread to finish
        print("✅ Application exited cleanly.")

```

In the `start` method, we start the continuous listening loop and the speech processing loop in separate threads. We then wait for the user to exit the application or press `Ctrl+C` to stop the application.

This completes our `ConversationManager` class. We have set up the speech recognition loop, processed user queries, and responded to the user's input.

#### 6. Create the main application

Finally, we create the `main.py` file, which will be the entry point of our application from the command line.

```python
from queue import Queue

from classes.ai import LlmManager
from classes.conversation import ConversationManager

if __name__ == "__main__":

    # Instantiate classes
    speech_queue = Queue()
    llm_manager = LlmManager()
    conversation_manager = ConversationManager(
        queue=speech_queue,
        llm_manager=llm_manager
    )

    # Start the conversation manager
    conversation_manager.start()
```

In the `main.py` file, we import the necessary classes and instantiate the `LlmManager` and `ConversationManager` classes. We then start the conversation manager to begin the conversation flow between the user and the AI voice agent.

This completes the setup of our AI voice agent. You can now run the `main.py` file to start the application and interact with the AI voice agent.

## Happy coding! 🚀

```bash
🐢 Arun Godwin Patel @ Code Creations
```
