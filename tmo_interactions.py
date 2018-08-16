# Topic Modeling with Sci-kit Learn Using Opinion Lab Data.
# Non-Negative Matrix Factorization (NMF) and Latent Dirichlet Allocation (LDA) algorithms.

import os
import os.path
import pandas as pd
import numpy as np
import warnings
import time
import nltk
import matplotlib.pyplot as plt
import pickle
# from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
# from nltk.stem.porter import PorterStemmer

warnings.filterwarnings('ignore')
#english_stemmer = nltk.stem.SnowballStemmer('english')
# pd.options.display.max_colwidth = 2000

# Setting Stopwords to be used
my_stopwords = nltk.corpus.stopwords.words('english')
# Adding other useless words to stopwords list:
my_stopwords.extend(['phone','number','customer','ticket','cust', 'ph','2017', '2018',
                     '2016','18','com'])

# n_components = 5
# no_top_words = 5
no_features =  1000


def output_file_name(model_name, brand):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    csv_file_path = './output/'
    csv_file_name = 'output_' + brand + '_' + model_name + '_' + timestr + '.csv'
    return csv_file_path + csv_file_name
    

def display_topics(model, feature_names, no_top_words):
    # Function to display topics extracted my a model:
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}: {}".format(topic_idx + 1,
                                    " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]] )))


def display_topics_documents(H, W, feature_names, documents, no_top_words, n_components):
    # Print out a numerical index as the topic name, the top words in the topic, the top ocuments in the topic.
    # The top words and top documents have the highest weights in the returned matrices:
    for topic_idx, topic in enumerate(H):
        print('Topic {}:'.format(topic_idx))
        print(' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        # print('\n')
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:n_components]
        for doc_index in top_doc_indices:
            print('-----')
            print(' {} '.format(doc_index) + documents[doc_index])
        print('\n')


def get_output_df(H, W, feature_names, n_components, no_top_words, complaints ):
    orig_col_names = complaints.columns.values
    out_df = pd.DataFrame(columns=orig_col_names)
    out_df = out_df.assign(Topic_Num = "")
    out_df = out_df.assign(Topic_Words = "")
    for topic_idx, topic in enumerate(H):
        this_topic = ' '.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        this_topic_num = topic_idx
        # print('Topic_num: {}, Topic: {}'.format(this_topic_num + 1, this_topic))
        top_doc_indices = np.argsort( W[:,topic_idx] )[::-1][0:n_components+10]
        for i in top_doc_indices:
            out_df.loc[i] = complaints.iloc[i] 
            out_df.set_value(i, 'Topic_Num', str(this_topic_num + 1))
            out_df.set_value(i, 'Topic_Words', this_topic)
            # out_df.loc[i]['Topic_Num','Topic'] = [this_topic_num, this_topic]
            
    return out_df

def get_complaints(dataset):
    complaints = dataset
    # Getting rid of short comments:
    complaints  = complaints[complaints['NOTES'].apply(lambda x: len(x) > 300)]
    return complaints

def main():

    n_components = 10

    no_top_words = 5

    # get dataset
    dataset = pickle.load(open('../datasets/result_df.pkl', 'rb'))
    dataset.dropna(axis=0,how='any',subset=['NOTES'],inplace=True)
    print(dataset['NOTES'].isna().any())

    # Print Number of rows for analysis:
    print('Number of rows to analyze: {}'.format(len(dataset)))
    print('\n')

    # plotting counts by rating
    # plot_counts_by_rating(dataset)
    # getting complaints only
    complaints = get_complaints(dataset)

    # Creating a list with all complaints/comments:
    documents = complaints["NOTES"].tolist()

    # Input for NMF and LDA Algorithms:
    #     1) Bag of Words Matrix with documents represented as rows and words  represented as columns
    #     2) the number of topics (k) that must be derived as a parameter
    # Output for NMF and LDA:
    #     1) the Documents-to-Topics matrix (W)
    #     2) The words-to-Topics matrix (H)

    # NMF model relies on linear algebra:
    # NMF is able to use tf-idf:
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=my_stopwords)
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    # Run NMF
    print('Initializing NMF Model ...')
    nmf_model = NMF(n_components=n_components,
                    random_state=1,
                    alpha=.1,
                    l1_ratio=.5,
                    init='nndsvd').fit(tfidf)
    nmf_W = nmf_model.transform(tfidf)
    nmf_H = nmf_model.components_
    print('Displaying topics extracted by NMF Model: ')
    display_topics(nmf_model, tfidf_feature_names, no_top_words)
    print('\n')
    
    # LDA is based on probabilistic graphical modeling
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model:
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=my_stopwords)
    tf = tf_vectorizer.fit_transform(documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    # Obtain the word to topics matrix (H) and the topics to documents matrix (W) from both the NMF and LDA algorithms.
    # Run LDA
    print('Initializing LDA Model ...')
    lda_model = LatentDirichletAllocation(n_components=n_components,
                                          max_iter=5,
                                          learning_method='online',
                                          learning_offset=50.,
                                          random_state=0).fit(tf)
    lda_W = lda_model.transform(tf)
    lda_H = lda_model.components_

    print('Displaying topics extracted by LDA Model: ')
    display_topics(lda_model, tf_feature_names, no_top_words)
    print('\n')

    #  # Printing Top Documents by Topic for each Model (NMF and LDA):
    # display_topics_documents(   nmf_H,
    #                             nmf_W,
    #                             tfidf_feature_names,
    #                             documents,
    #                             no_top_words,
    #                             n_components
    # 
    #                         )

    my_nmf_df = get_output_df(nmf_H, 
                              nmf_W, 
                              tfidf_feature_names, 
                              n_components, 
                              no_top_words,
                              complaints) 

    csv_file_name = output_file_name(model_name='NMF', brand='TMO')
    print('Exporting NMF output data to csv file {} ...'.format(csv_file_name))                                                        
    my_nmf_df.to_csv(csv_file_name)
    
    # display_topics_documents(   lda_H,
    #                              lda_W,
    #                              tf_feature_names,
    #                              documents,
    #                              no_top_words,
    #                              n_components)

    my_lda_df = get_output_df(lda_H, 
                              lda_W, 
                              tf_feature_names, 
                              n_components, 
                              no_top_words,
                              complaints )

    csv_file_name = output_file_name(model_name='LDA', brand='TMO')
    print('Exporting LDA output data to csv file {} ...'.format(csv_file_name))
    my_lda_df.to_csv(csv_file_name)


if __name__ == '__main__':
    main()
