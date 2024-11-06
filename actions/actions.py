from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionCheckPunSentence(Action):
    def name(self) -> Text:
        return "action_check_pun_sentence"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
          
        pun_sentence = tracker.latest_message.get('text')

        is_pun_detected = self.detect_pun(pun_sentence)

        if is_pun_detected:
            dispatcher.utter_message(text="Pun detected! Parsing the pun...")
        else:
            dispatcher.utter_message(text="No pun detected. Please try another sentence.")
        return [SlotSet("pun_exists", is_pun_detected)]


    def detect_pun(self, sentence: Text) -> bool:
        return True
    
class ActionCheckMatchingPunWord(Action):
    def name(self) -> Text:
        return "action_detect_matching_pun_word"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
          
        pun_word = tracker.latest_message.get('text')

        matched_word = self.detect_similar_word(pun_word)

        return [SlotSet("matched_pun_word", matched_word)]

        

    def detect_similar_word(self, sentence: Text) -> bool:

        return True