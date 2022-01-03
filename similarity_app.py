import streamlit as st
import numpy as np
import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_transformers import SentenceTransformer, util
from simple_elmo import ElmoModel

import gensim
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

elmo_model = ElmoModel()
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 

st.set_page_config(
   page_title="Text Similarity and Embedding Techniques Demo",
   page_icon="🛠",
   layout="centered",
   initial_sidebar_state="expanded",
   menu_items={
         'About': 'How bla bla bla',
            }
)

@st.experimental_singleton
def prepare_models():
    elmo_model = ElmoModel()
    elmo_model.load("/content/elmo")
    transformer_model = SentenceTransformer('stsb-roberta-large')
    USE_model = hub.load(module_url)
    nlp = spacy.load('en_core_web_md')
    return transformer_model, USE_model, elmo_model, nlp

transformer_model, USE_model, elmo_model, nlp = prepare_models()

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
               row.append(1/np.exp((euclidean_distances(docs[i].vector, docs[j].vector))))
        similarity.append(row)
    return similarity        

# elmo
def elmo_similarity(sentence):
    elmo_vectors = elmo_model.get_elmo_vectors(sentence, layers="average")
    elmo_vectors = np.moveaxis(elmo_vectors, [0, 1, 2], [1, 0, 2])
    # get word embeddings
    word_vectors = []
    for word in sentence.split(" "):
        start_idx = sentence.index(word)
        l = len(word)
        # print(sentence[start_idx: start_idx+l])
        word_vectors.append(np.sum(elmo_vectors[0][start_idx:start_idx + l], axis = 0)/l)

    #calculate similarity
    similarity = []
    for i in range(len(word_vectors)):
        row = []
        for j in range(len(word_vectors)):
            row.append(cosine_similarity(word_vectors[i].reshape(1, -1), word_vectors[j].reshape(1, -1))[0][0])
        similarity.append(row)
        return similarity

# Doc2Vec
def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

def doc2vec_cosine(sentences, flag):
    tokenized_sent = []
    for s in sentences:
        tokenized_sent.append(word_tokenize(s.lower()))
    training_data = list(tagged_document(tokenized_sent))
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=10)
    doc2vec_model.build_vocab(training_data)
    doc2vec_model.train(training_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
    vectors = [doc2vec_model.infer_vector([word for word in sent]).reshape(1,-1) for sent in sentences]
    if flag == 0:
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    else:
        similarity = []
        for i in range(len(sentences)):
            row = []
            for j in range(len(sentences)):
                row.append(cosine_similarity(vectors[i], vectors[j])[0][0])
            similarity.append(row)
        return similarity


# USE
def USE_cosine(sentences, flag):
    embeddings = USE_model(sentences)
    if flag == 0:
        return cosine_similarity(embeddings)[0][1]
    else:
        return cosine_similarity(embeddings)

# Sentence Transformer
def transformer_cosine(sentences, flag):
    embeddings = transformer_model.encode(sentences, convert_to_tensor=True)
    if flag == 0:
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    else:
        similarity = []
        for i in range(len(sentences)):
            row = []
            for j in range(len(sentences)):
                row.append(util.pytorch_cos_sim(embeddings[i], embeddings[j]).item())
            similarity.append(row)
        return similarity

# helper methods
tokenize = spacy.load('en_core_web_sm', disable=['parser', 'ner',
                                            'tok2vec', 'attribute_ruler'])

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
    sns.heatmap(df, cmap=cmap)
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
        sentence_3 = st.text_input('Sentence 3', 'Investors unfazed by correction as crypto funds see $154 million inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 3')
        sentence_4 = st.text_input('Sentence 4', 'Bitcoin, Ethereum prices continue descent, but crypto funds see inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 4')
        sentence_5 = st.text_input('Sentence 5', 'Investors unfazed by correction as crypto funds see $154 million inflows', max_chars = 100, 
                        help = 'Sentence To Test',
                        placeholder = 'Sentence 5')
        return [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]

sentence_emb_methods = {'Doc2Vec': doc2vec_cosine,
'Universal Search Encoder' : USE_cosine,
'Sentence Transformers': transformer_cosine}

word_emb_methods = {'Bag Of Words':count_vectorizer,
'TF-IDF' : tfid_vectorizer, 'Word2Vec' : word2vec}

st.title('Text Similarity and Embedding Techniques Demo')
# st.subheader('No code live demo for News API. This is just a small part of entire functionality,\
# sign up to unock the full potential')

st.subheader('Simlarity Measures')

measure = st.radio(
     "What's similarity metric do you want to try?",
     ('Jaccard', 'Euclidean', 'Cosine'))

if measure == 'Jaccard':
    no_sent = st.radio(
     "Pair of sentences or five sentences?",
     ('Pair', 'Five'))

    sentences = get_sentences(2 if no_sent == 'Pair' else 5)

    docs = [tokenize(sentence) for sentence in sentences]
    tokens = []
    
    for doc in docs:
        temp = []
        for token in doc:
            temp.append(token.lemma_)
        tokens.append(temp)    
    tokens_no_punc = list(map(remove_punctuations, tokens))
    similarity = calc_jaccard(tokens_no_punc)

    # flag for single value vs matrix of similarities
    if no_sent == 'Pair':
        st.write(round(similarity[0][1],3))
    else:
        st.write(create_heatmap(similarity, sentences))

if measure == 'Euclidean':
    no_sent = st.radio(
     "Pair of sentences or five sentences?",
     ('Pair', 'Five'))
    sentences = get_sentences(2 if no_sent == 'Pair' else 5)

    st.subheader('Embedding Type')

    emb_type = st.radio(
     "What's embedding type do you want to try?",
     ('Bag Of Words', 'TF-IDF', 'Word2Vec'))

    similarity = word_emb_methods[emb_type](sentences,'euclidean')

    if no_sent == 'Pair':
        st.write(round(similarity[0][1],3))
    else:
        st.write(create_heatmap(similarity, sentences))

if measure == 'Cosine':
    st.subheader('Embedding Type')
    emb_type = st.radio(
        "What's embedding type do you want to try?",
        ('Traditional Embeddings', 'Contextual Word Embeddings', 'Sentence Embeddings'))

    st.subheader('Sentence(s) To Test')

    if emb_type == 'Contextual Word Embeddings':
        
        sentence = st.text_input('Sentence', 'After stealing gold from the bank vault, the bank robber was seen fishing on the river bank.', max_chars = 100, 
                        help = 'Enter a sentence that uses words in multiple contexts',
                        placeholder = 'Enter a sentence that uses words in multiple contexts')
        similarity = elmo_similarity(sentence)
        st.wrte(create_heatmap(similarity, sentence, words = True))

    if emb_type == 'Traditional Embeddings':
        no_sent = st.radio(
        "Pair of sentences or five sentences?",
        ('Pair', 'Five'))
        sentences = get_sentences(2 if no_sent == 'Pair' else 5)

        st.subheader('Embedding Type')

        emb_type = st.radio(
        "What's embedding type do you want to try?",
        ('Bag Of Words', 'TF-IDF', 'Word2Vec'))

        similarity = word_emb_methods[emb_type](sentences)

        if no_sent == 'Pair':
            st.write(round(similarity[0][1],3))
        else:
            st.write(create_heatmap(similarity, sentences))

    if emb_type == 'Sentence Embeddings':
        no_sent = st.radio(
        "Pair of sentences or five sentences?",
        ('Pair', 'Five'))
        sentences = get_sentences(2 if no_sent == 'Pair' else 5)

        st.subheader('Embedding Type')

        emb_type = st.radio(
        "What's embedding type do you want to try?",
        ('Doc2Vec', 'Universal Search Encoder', 'Sentence Transformers'))

        similarity = sentence_emb_methods[emb_type](sentences,0 if len(sentences) == 2 else 1)

        if no_sent == 'Pair':
            st.write(round(similarity[0][1],3))
        else:
            st.write(create_heatmap(similarity, sentences))
