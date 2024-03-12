from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import string, re
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
import pickle
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk import bigrams 
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
app = Flask(__name__)
recipes = pd.read_csv(r"df_recipes_cleaned.csv")

# Load pre-trained NMF model
def load_nmf_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model
def commatokenizer(text):
    return text.split(', ')
# Load pre-trained vectorizer
def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Load document-topic matrix
def load_doc_topic(doc_topic_path):
    with open(doc_topic_path, 'rb') as f:
        doc_topic = pickle.load(f)
    return doc_topic

# Load pre-trained models and data
nmf_model = load_nmf_model(r"NMF_model.sav")
vectorizer = load_vectorizer(r'vectorizer.pkl')
doc_topic = np.load(r'doc_topic.npy', allow_pickle=True)

# Placeholder preprocessing functions
def preprocessor(text):
    ingredlist = []
    for ingred in  text.split(', '):
        ingred = re.sub('\w*\d\w*', ' ', ingred)  # Remove any words containing digits 
        ingred = ingred.replace('"', '').replace("'", '').replace('& ', '').replace('-','')   
        ingred = re.sub('[%s]' % re.escape(string.punctuation), ' ', ingred)  # Remove punctuation
        ingred = ingred.lower().strip()        
#         new_list = []
#         for word in ingred.split():
#             new_list.append(singularizer(word))
#         ingred = ' '.join(new_list)
        ingredlist.append(ingred)        
    return ', '.join(ingredlist) 
def is_noun(word):
    nouns = {'NN','NNS', 'NNP', 'NNPS','NOUN', 'PROPN', 'NE', 'NNE', 'NR'}
    pos = nlp(word)[0].tag_ 
    if pos in nouns:
        return True
    return False
def word_singularizer(word):
    nlp_word = nlp(word)[0]
    lemma = nlp_word.text
    if nlp_word.tag_ in {"NNS", "NNPS"}:
            lemma = nlp_word.lemma_
    return lemma
def get_nouns(text):
    tokens = RegexpTokenizer(r'\w+').tokenize(text)
    nounlist = [word_singularizer(word) for word in tokens if is_noun(word)]
    return ', '.join(nounlist) 

def text_singularizer(text):
    ingredlist = []
    for ingred in  text.split(', '):
        new_list = []
        for word in ingred.split():
            new_list.append(word_singularizer(word))
        ingred = ' '.join(new_list)
        ingredlist.append(ingred)        
    return ', '.join(ingredlist) 

def commatokenizer(text):
    return text.split(', ')

def mytokenizer(combinedlist):
    ingredlist = combinedlist[0].split(', ')
    nounlist = combinedlist[1].split(', ')
    ingredlist = combinedlist[0].split(', ')
    bigramlist = []
    for ingred in ingredlist:
        bigrms = [bi for bi in bigrams(ingred.split())]
        for bi in bigrms:
            if (bi[0] in nounlist) or (bi[1] in nounlist):
                bigramlist.append(' '.join((bi[0], bi[1])))
   
    return ', '.join(bigramlist + nounlist)

# Function to tokenize user input
def user_tokenize(ingreds):
    ingreds = preprocessor(ingreds)
    nouns = get_nouns(ingreds)
    ingreds = text_singularizer(ingreds)
    ingredscombined = [ingreds, nouns]
    ingredstokenized = mytokenizer(ingredscombined)
    return ingredstokenized

# Function to get recipe recommendations
def get_recommendations(useringred):
    # Split input string by comma and convert to lowercase
    user_tokens = [token.strip().lower() for token in useringred.split(",")]

    matching_recipes = []

    # Iterate through recipes to find matching ones
    for index, recipe in recipes.iterrows():
        ingredients_tokenized = recipe['IngredientsTokenized']
        if isinstance(ingredients_tokenized, str):  # Check if it's a string
            # Tokenize and convert ingredients to lowercase
            ingredients = [ingredient.strip().lower() for ingredient in ingredients_tokenized.split(", ")]
            # Check if at least five user tokens match ingredients
            matched_count = sum(1 for token in user_tokens if token in ingredients)
            if matched_count >= 5:
                matching_recipes.append(index)  # Append index of matching recipe

    return matching_recipes


@app.route('/recommendations', methods=['POST'])
def recommendations():
    useringreds = request.json.get('ingredients', '')
    if not useringreds:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    recommendations_indices = get_recommendations(useringreds)
    
    recommended_recipes = []
    # Iterate through matching recipe indices
    for index in recommendations_indices:
        recipe_dict = recipes.iloc[index].to_dict()  # Access recipe by index
        recommended_recipes.append(recipe_dict)
    
    return jsonify(recommended_recipes), 200

sia = SentimentIntensityAnalyzer()

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data.get('text', '')

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    return jsonify(sentiment_scores)


if __name__ == '__main__':
    app.run(debug=True)

