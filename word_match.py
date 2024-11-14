import nltk
from nltk.corpus import cmudict, words, wordnet as wn
import numpy as np
import itertools
 
# Download necessary resources from nltk
nltk.download('cmudict')
nltk.download('words')
nltk.download('wordnet')
 
# Load CMU Pronouncing Dictionary and words list
cmu_dict = cmudict.dict()
word_list = set(words.words())
 
# Extended feature-based representation for phonemes
# Features: [voicing, nasal, plosive, fricative, place]
phoneme_features = {
    'P': [0, 0, 1, 0, 3],  # voiceless, non-nasal, plosive, labial
    'B': [1, 0, 1, 0, 3],  # voiced, non-nasal, plosive, labial
    'M': [1, 1, 0, 0, 3],  # voiced, nasal, non-plosive, labial
    'N': [1, 1, 0, 0, 5],  # voiced, nasal, non-plosive, alveolar
    'T': [0, 0, 1, 0, 5],  # voiceless, non-nasal, plosive, alveolar
    'D': [1, 0, 1, 0, 5],  # voiced, non-nasal, plosive, alveolar
    'K': [0, 0, 1, 0, 7],  # voiceless, non-nasal, plosive, velar
    'G': [1, 0, 1, 0, 7],  # voiced, non-nasal, plosive, velar
    'F': [0, 0, 0, 1, 2],  # voiceless, non-nasal, fricative, labiodental
    'V': [1, 0, 0, 1, 2],  # voiced, non-nasal, fricative, labiodental
    'S': [0, 0, 0, 1, 5],  # voiceless, non-nasal, fricative, alveolar
    'Z': [1, 0, 0, 1, 5],  # voiced, non-nasal, fricative, alveolar
    'SH': [0, 0, 0, 1, 6],  # voiceless, non-nasal, fricative, postalveolar
    'ZH': [1, 0, 0, 1, 6],  # voiced, non-nasal, fricative, postalveolar
    'CH': [0, 0, 1, 1, 6],  # voiceless, non-nasal, affricate, postalveolar
    'JH': [1, 0, 1, 1, 6],  # voiced, non-nasal, affricate, postalveolar
    'L': [1, 0, 0, 0, 5],  # voiced, non-nasal, lateral approximant, alveolar
    'R': [1, 0, 0, 0, 5],  # voiced, non-nasal, approximant, alveolar
    'Y': [1, 0, 0, 0, 8],  # voiced, non-nasal, approximant, palatal
    'W': [1, 0, 0, 0, 1],  # voiced, non-nasal, approximant, bilabial
    'HH': [0, 0, 0, 0, 8],  # voiceless, non-nasal, glottal fricative
    'NG': [1, 1, 0, 0, 7],  # voiced, nasal, non-plosive, velar
    'TH': [0, 0, 0, 1, 4],  # voiceless, non-nasal, fricative, dental
    'DH': [1, 0, 0, 1, 4],  # voiced, non-nasal, fricative, dental
    'AW': [1, 0, 0, 0, 9],  # voiced, diphthong, back
    'AY': [1, 0, 0, 0, 9],  # voiced, diphthong, front
    'OY': [1, 0, 0, 0, 9],  # voiced, diphthong, round front
    'EH': [1, 0, 0, 0, 5],  # voiced, short vowel, front
    'IH': [1, 0, 0, 0, 5],  # voiced, short vowel, high front
    'AA': [1, 0, 0, 0, 5],  # voiced, open vowel, back
    'AE': [1, 0, 0, 0, 5],  # voiced, front open vowel
    'UH': [1, 0, 0, 0, 7],  # voiced, high back vowel
    'OO': [1, 0, 0, 0, 7],  # voiced, round back vowel
}
 
# Function to calculate feature distance between two phonemes
def feature_distance(phoneme1, phoneme2):
    features1 = phoneme_features.get(phoneme1)
    features2 = phoneme_features.get(phoneme2)
 
    if features1 is None or features2 is None:
        return 1  # Default high cost for unknown phonemes
 
    # Use Hamming distance or L1 distance to calculate feature difference
    return sum(abs(f1 - f2) for f1, f2 in zip(features1, features2)) / len(features1)
 
# Weighted edit distance using feature vectors
def weighted_edit_distance(seq1, seq2):
    len1, len2 = len(seq1), len(seq2)
    dp = np.zeros((len1 + 1, len2 + 1))
 
    # Initialize DP table
    for i in range(len1 + 1):
        dp[i][0] = i  # Cost of deletion
    for j in range(len2 + 1):
        dp[0][j] = j  # Cost of insertion
 
    # Fill the DP table with generalized weighted costs
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if the same
            else:
                substitution_cost = feature_distance(seq1[i - 1], seq2[j - 1])
                dp[i][j] = min(dp[i - 1][j] + 1,  # Deletion
                               dp[i][j - 1] + 1,  # Insertion
                               dp[i - 1][j - 1] + substitution_cost)  # Substitution
 
    return dp[len1][len2]
 
# Function to remove stress markers from CMU transcriptions
def normalize_phoneme(phoneme):
    return ''.join([char for char in phoneme if not char.isdigit()])
 
# Function to normalize transcription by removing stress markers
def normalize_transcription(transcription):
    return [normalize_phoneme(phoneme) for phoneme in transcription]
 
# Function to get the phonetic transcription of a word or phrase
def get_phonetic_transcription(text):
    words = text.lower().split()
    transcription = []
 
    for word in words:
        if word in cmu_dict:
            # Take the first pronunciation variant
            transcription.extend(cmu_dict[word][0])
 
    return transcription
 
from itertools import product
 
# Function to find similar-sounding words for a single word
def find_similar_sounding_words(target_word, max_distance=2):
    similar_words = []
    target_transcription = get_phonetic_transcription(target_word)
    if not target_transcription:
        return similar_words
    for word, transcriptions in cmu_dict.items():
        word_lower = word.lower()
        # Skip the exact match
        if word_lower == target_word.lower():
            continue
 
        # Only check valid English words
        if word_lower not in word_list or not wn.synsets(word_lower):
            continue
 
        for transcription in transcriptions:
            # Calculate weighted edit distance between target and candidate transcription
            distance = weighted_edit_distance(target_transcription, transcription)
 
            # If within the acceptable range, add to similar words
            if distance <= max_distance:
                similar_words.append(word)
                break  # Stop after finding one similar transcription
    return similar_words
 
# Function to find phonetically similar phrases and single words
def find_similar_sounding_words_and_phrases(target_text, max_distance=2):
    target_words = target_text.lower().split()
    similar_words_list = []
    all_similar_phrases = {}
    print(target_text)
    print(target_words)
    # Generate similar-sounding words for each word in the target phrase
    for word in target_words:
        similar_words = find_similar_sounding_words(word, max_distance=max_distance)
        # Include original word to check for exact matches in combinations
        similar_words.append(word)
        similar_words_list.append(similar_words)
 
    # Generate possible phrase combinations from similar-sounding words
    for phrase_tuple in product(*similar_words_list):
        phrase = ' '.join(phrase_tuple)
        phrase_transcription = get_phonetic_transcription(phrase)
 
        if not phrase_transcription:
            continue  # Skip if no transcription
 
        # Calculate distance between target transcription and current phrase transcription
        target_transcription = get_phonetic_transcription(target_text)
        distance = weighted_edit_distance(target_transcription, phrase_transcription)
        if distance <= max_distance and phrase != target_text.lower():
            all_similar_phrases[phrase] = distance
 
    # Find single words similar to target phrase
    similar_single_words = {}
    target_transcription = get_phonetic_transcription(target_text)
    for word, transcriptions in cmu_dict.items():
        word_lower = word.lower()
        if word_lower in target_text.lower():
            continue  # Skip exact match
        if word_lower not in word_list or not wn.synsets(word_lower):
            continue  # Check if word exists in WordNet
 
        for transcription in transcriptions:
            # Calculate distance between target transcription and each dictionary word transcription
            distance = weighted_edit_distance(target_transcription, transcription)
            if distance <= max_distance:
                similar_single_words[word] = distance
                break
 
    # Sort results by edit distance
    all_similar_phrases = dict(sorted(all_similar_phrases.items(), key=lambda item: item[1]))
    similar_single_words = dict(sorted(similar_single_words.items(), key=lambda item: item[1]))
 
    return similar_single_words, all_similar_phrases
 
# Example usage
target_text = "whale"
similar_single_words, similar_phrases = find_similar_sounding_words_and_phrases(target_text, max_distance=2)
 
# Merge word list and phrase list
similar_words = similar_single_words.copy()
similar_words.update(similar_phrases)
 
# Sort word list by similarity score
similar_words = dict(sorted(similar_words.items(), key=lambda item: item[1]))
 
# Find the closest matches with the lowest score
if similar_words:
    min_distance = min(similar_words.values())
    closest_words_or_phrases = {word: dist for word, dist in similar_words.items() if dist == min_distance}
else:
    closest_words_or_phrases = {}
 
# Print results
print(f"Closest words or phrases with similar pronunciation to '{target_text}': {closest_words_or_phrases}")
print(list(closest_words_or_phrases.keys()))