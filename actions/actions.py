from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import AllSlotsReset

from pun_detection.detect_puns import chatbot_response


class ActionExplainJoke(Action):
    def name(self) -> Text:
        return "action_explain_joke"
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        joke = tracker.get_slot("joke")
        
        response = chatbot_response(joke)
                
        if response:
            dispatcher.utter_message(text=response)
            
        else:
            dispatcher.utter_message(response='utter_explain_joke')
            
        return [AllSlotsReset()] 
        
        
        
        