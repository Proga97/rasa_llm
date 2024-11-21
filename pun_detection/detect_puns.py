from guidance import models, gen, assistant, user, system, select, substring
from pun_detection.find_homophones import get_homophones

from nltk.corpus import wordnet as wn

import os

# Chatbot response
system_should_explain_joke = 'Is the user asking you to explain a joke or pun?'
system_extract_pun = '''Which words in the user message make up the pun? 
Example: 
Q: Please explain this joke: "I used to be a baker, but I couldn\'t make enough dough."
A: The pun word(s) is: dough'''
system_pun_type = '''Is the  humor in the pun based on "sound" or "meaning"?
Example:
The pun "What do you call a fake noodle? An impasta." is based on "sound", because "impasta" sounds like "imposter".
The pun "My wife told me to stop impersonating a flamingo. I had to put my foot down." is based on "meaning", because "put my foot down" is an idiom meaning to assert authority.'''

# Explain joke
system_explain_joke = 'Explain the humor in the joke or pun.'

error_message = None

curr_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(curr_dir, 'Llama-3.2-3B-Instruct-Q6_K_L.gguf')

llama = models.LlamaCpp(path, echo=False, n_ctx=4096)

def explain_joke(lm):
    with system():
        llm = lm + system_explain_joke
    with assistant():
        llm = gen('explain_joke')
        if 'explain_joke' in llm:
            return llm['explain_joke']
        else:
            return error_message
    

def handle_homograph(lm, pun):
    
    synsets = wn.synsets(pun)
    definitions = [syn.definition() for syn in synsets]
    
    if len(definitions) < 2:
        return explain_joke(lm)
        
    elif len(definitions) == 2:
        def1, def2 = definitions
    
    else:
        with system():
            llm = lm + 'The pun word "' + pun + '" plays on multiple meanings. Of the following dictionary definitions, which two explain the pun? \n - ' + '\n - '.join(definitions)
            
        with assistant():
            llm += 'The humor plays off of the pun "' + pun + '" having two distinct meanings in this context. First, the definition "' + select(definitions, name = 'def1') + '"'
            if 'def1' in llm:
                def1 = llm['def1']
                definitions.remove(def1)
                llm += '. Second, "' + pun + '" also has the definition: "' + select(definitions, name = 'def2')
                if 'def2' in llm:
                    def2 = llm['def2']
                else:
                    return explain_joke(lm)
            else:
                return explain_joke(lm)
        
    with assistant():
        lm += 'The pun "' + pun + '" is a homograph. It plays off the two meanings of the word "' + pun + '": (1) ' + def1 + ' and (2) ' + def2 + '. ' + gen(stop='.', name='explanation')
        return 'The pun "' + pun + '" is a homograph. It plays off the two meanings of the word "' + pun + '": (1) ' + def1 + ' and (2) ' + def2 + '. ' + (lm['explanation'] + '.' if 'explanation' in lm else '')
    
def handle_homophone(lm, pun):
    
    cleansed_pun = pun.replace('-', ' ').lower()
    cleansed_pun = ''.join([c for c in cleansed_pun if c.isalpha() or c.isspace()])
    
    similar_phones = get_homophones(cleansed_pun)
    
    if len(similar_phones) == 0:
        return explain_joke(lm)
    
    elif len(similar_phones) == 1:
        with assistant():
            homophone = similar_phones[0]
                                
    else:
        with system():
            llm = lm + 'The pun word "' + pun + '" sounds like multiple word(s). Which one is the pun? \n - ' + '\n - '.join(similar_phones)
        
        with assistant():
            llm += 'The pun "' + pun + '" is a pun because it sounds like the following word(s): ' + select(similar_phones, name = 'homophone')
        
        if 'homophone' not in llm:
            return explain_joke(lm)
                
        homophone = llm['homophone']
        
    with assistant():
        lm += 'The joke relies on the homophone "' + pun + '", which sounds like "' + homophone + '". ' + gen(stop='.', name='explanation')
        return 'The joke relies on the homophone "' + pun + '", which sounds like "' + homophone + '". ' + (lm['explanation'] + '.' if 'explanation' in lm else '')

def chatbot_response(joke):   
    
    with user():
        lm = llama + 'Please explain this joke: "' + joke + '"'
                
    with system():
        lm += system_extract_pun
        
    with assistant():
        lm += 'The pun word(s) is: ' + substring(joke, name='pun')
    
    if 'pun' in lm:
        pun = lm['pun'].strip()
        
        with system():
            lm += system_pun_type
        
        with assistant():
            lm += select(['sound', 'meaning'], name='pun_type')
        
        if 'pun_type' in lm:
            
            if lm['pun_type'] == 'sound':
                return handle_homophone(lm, pun)
                                    
            else:
                return handle_homograph(lm, pun)
    
    with assistant():
        lm += gen('explanation')
        
    if 'explanation' in lm:
        return lm['explanation']

    else:
        return error_message
    