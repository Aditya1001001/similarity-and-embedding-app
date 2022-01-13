import streamlit as st
import numpy as np
import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
   page_title="Text Similarity and Embedding Techniques Demo",
   page_icon="🛠",
   layout="centered",
   initial_sidebar_state="expanded",
   menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": None
            }
)

@st.experimental_singleton
def prepare_models():

    nlp = spacy.load('en_core_web_md')
    tokenize = spacy.load('en_core_web_sm', disable=['parser', 'ner',
                                            'tok2vec', 'attribute_ruler'])
    return  nlp, tokenize

nlp, tokenize = prepare_models()

# count vectorizer
def count_vectorizer(sentences, metric = 'cosine'):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    arr = X.toarray()
    if metric == 'cosine':
        return cosine_similarity(arr)
    else:
        return 1/np.exp((euclidean_distances(arr)))

# tfidf vectorizer
def tfid_vectorizer(sentences, metric = 'cosine'):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    arr = X.toarray()
    if metric == 'cosine':
        return cosine_similarity(arr)
    else:
        return 1/np.exp((euclidean_distances(arr)))

# word2vec
def word2vec(sentences, metric = 'cosine'):
    docs = [nlp(sentence) for sentence in sentences]
    similarity = []
    for i in range(len(docs)):
        row = []
        for j in range(len(docs)):
            if metric == 'cosine':
              row.append(docs[i].similarity(docs[j]))
            else:
               row.append(1/np.exp((euclidean_distances(docs[i].vector.reshape(1, -1), docs[j].vector.reshape(1, -1))[0][0])))
        similarity.append(row)
    return similarity    

# helper methods
def remove_punctuations(normalized_tokens):
    punctuations=['?',':','!',',','.',';','|','(',')','--']
    for word in normalized_tokens:
        if word in punctuations:
            normalized_tokens.remove(word)
    return normalized_tokens

def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)

def calc_jaccard(sentences):
  similarity = []
  for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
      row.append(jaccard_similarity(sentences[i], sentences[j]))
    similarity.append(row)
  return similarity

def create_heatmap(similarity, sentences, words = False, cmap = "YlGnBu"):
    if words:
        labels = sentences.lower().split(" ")
    else:
        labels = [sentence[:20] for sentence in sentences]
    df = pd.DataFrame(similarity)
    df.columns = labels
    df.index = labels
    fig, ax = plt.subplots(figsize=(7,7))
    sns.heatmap(df, cmap=cmap, vmin=0, vmax=1)
    return fig

# StreamLit App        
def get_sentences(n):
    if n == 2:
        sentence_1 = st.text_input('Sentence 1', 'Investors unfazed by correction as crypto funds see $154 million inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 1')
        sentence_2 = st.text_input('Sentence 2', 'Bitcoin, Ethereum prices continue descent, but crypto funds see inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 2')
        return [sentence_1, sentence_2]
    else:
        sentence_1 = st.text_input('Sentence 1', 'Investors unfazed by correction as crypto funds see $154 million inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 1')
        sentence_2 = st.text_input('Sentence 2', 'Bitcoin, Ethereum prices continue descent, but crypto funds see inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 2')
        sentence_3 = st.text_input('Sentence 3', 'The surge in euro area inflation during the pandemic: transitory but with upside risks', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 3')
        sentence_4 = st.text_input('Sentence 4', "Inflation: why it's temporary and raising interest rates will do more harm than good", max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 4')
        sentence_5 = st.text_input('Sentence 5', 'Will Cryptocurrency Protect Against Inflation?', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 5')
        return [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]

word_emb_methods = {'Bag Of Words':count_vectorizer,
'TF-IDF' : tfid_vectorizer, 'Word2Vec' : word2vec}

def get_embedding_and_sentences(sentences, metric):

    emb_type = st.sidebar.radio(
     "What's embedding type do you want to use to calculate similarity?",
     ('Bag Of Words', 'TF-IDF', 'Word2Vec'))

    return word_emb_methods[emb_type](sentences, metric), no_sent


st.title('Text Similarity Measures Live Demo')
st.write('Text Similarity Score measures how alike or different two text \
    documents are. As simple as the idea might sound, many Natural Language \
    Processing applications use similarity behind the scenes. This app enables \
    you to play around with the three most common text similarity measures \
    along with a few embedding methods.')
st.write('To learn more about text similarity \
    and embedding methods, read our in-depth article \
    [here](https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python).')

measure = st.sidebar.radio(
     "What's similarity metric do you want to try?",
     ('Jaccard', 'Euclidean', 'Cosine'))

if measure == 'Jaccard':
    no_sent = st.sidebar.radio(
     "Pair of sentences or five sentences?",
     ('Pair', 'Five'))
    sentences = get_sentences(2 if no_sent == 'Pair' else 5)

    with st.spinner("Calculating Similarity"):
        docs = [tokenize(sentence) for sentence in sentences]
        tokens = []
        
        for doc in docs:
            temp = []
            for token in doc:
                temp.append(token.lemma_)
            tokens.append(temp)    
        tokens_no_punc = list(map(remove_punctuations, tokens))
        similarity = calc_jaccard(tokens_no_punc)
        st.write("Similarity Scores lie in the range of 0 to 1. \
             Here's the pairwise similarity score(s) of the above sentences:")
        if no_sent == 'Pair':
            st.subheader("Similarity")
            st.metric(" ", round(similarity[0][1],3), delta=None, delta_color="normal")
        else:
            st.write(create_heatmap(similarity, sentences))
else:
    no_sent = st.sidebar.radio(
     "Pair of sentences or five sentences?",
     ('Pair', 'Five'))
    sentences = get_sentences(2 if no_sent == 'Pair' else 5)

    with st.spinner("Calculating Similarity"):
        similarity, no_sent= get_embedding_and_sentences(sentences, measure.lower())
        st.write("Similarity Scores lie in the range of 0 to 1. \
            Here's the pairwise similarity score(s) of the above sentences:")
        if no_sent == 'Pair':
            st.subheader("Similarity")
            st.metric(" ", round(similarity[0][1],3), delta=None, delta_color="normal")
        else:
            st.write(create_heatmap(similarity, sentences))