import json
import time
import pyttsx3
import threading

import speech_recognition as sr


class ConversationManager:

    def __init__(self, queue, llm_manager):
        self.stop_listening = False
        self.energy_threshold = 5000
        self.speech_queue = queue
        self.llm_manager = llm_manager

    # Function to start the conversation manager
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

    # Function to continuously listen for speech in the background
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

    # Function to process the speech events
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

    # Function to provide text-to-speech feedback
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
