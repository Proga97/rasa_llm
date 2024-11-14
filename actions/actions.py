import random
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from open_ai_detection import detect_pun_word,detect_pun_word_match
from word_match import find_similar_sounding_words_and_phrases


class ActionCheckPunSentence(Action):
    def name(self) -> Text:
        return "action_check_pun_sentence"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        pun_sentence = tracker.get_slot("pun_sentence")
        if pun_sentence is None:
            dispatcher.utter_message(
                text="It seems you haven't provided a pun yet. Can you please try again?"
            )
            return []
        print("Pun sentence = " + "   "  + pun_sentence or "pun slot empty")
        is_pun_detected,pun_word  = self.detect_pun(pun_sentence)
        print("is_pun_detected = " + "   "  + str(is_pun_detected or "true/false") + "   " + str(pun_word or "no pun word foundzz" + "   " ))
        # if is_pun_detected:
        #     dispatcher.utter_message(text="Pun detected! Parsing the pun...")
        # else:
        #     dispatcher.utter_message(text="No pun detected. Please try another sentence.")

        return [
        SlotSet("pun_detected", is_pun_detected),
        SlotSet("pun_word", pun_word)]


    def detect_pun(self, sentence: str):

        ############# Alex's pun word detection algo goes here;

        # x = random.choice([True, False]) or True
        # return  x  , "pun_word" if x else None
        print(sentence)
        text = detect_pun_word(sentence)
        x = text.split(",")
        print("gpt ans = "  + x[0] == 'True' + x[1])

        #################### Alex's pun word detection algo goes here;

        return x[0],x[1]
    
class ActionCheckMatchingPunWord(Action):
    def name(self) -> Text:
        return "action_detect_matching_pun_word"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        pun_word = tracker.get_slot("pun_word")
        print("Pun matcherr " + str(pun_word or "missing word"))

        is_pun__match_detected ,matched_word  = self.detect_similar_word(pun_word)
        
        # if is_pun__match_detected:
        #     dispatcher.utter_message(text="Pun matched!")
        #     dispatcher.utter_message(text="The matched word is " + matched_word)
        # else:
        #     dispatcher.utter_message(text="Oops couldn't find the matching pun word. Please try another sentence.")
        return [SlotSet("pun__match_detected", (is_pun__match_detected or False)),
        SlotSet("matched_pun_word", matched_word)]

        

    def detect_similar_word(self, sentence: str):
        ############# Hassan and Denison's pun word detection algo goes here;
        # x = random.choice([True, False])
        # return [ x , "matched_pun_word" if x else None]

        # text = detect_pun_word_match(sentence)
        text = find_similar_sounding_words_and_phrases(sentence)
        # x = text.split(",")
        print(text)
        # print("gpt ans = "  + x[0] == 'True' + x[1])

        ############# Hassan and Denison's pun word detection algo goes here;

        # return x[0],x[1]
    