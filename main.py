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
