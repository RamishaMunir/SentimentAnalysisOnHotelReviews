import matplotlib.pyplot as plt, mpld3
from matplotlib.pyplot import hist
import pandas as pd
import seaborn as sns
from empath import Empath
import sqlalchemy
from textblob import TextBlob
from sqlalchemy import create_engine
import mysql.connector as SQLC

import numpy as np
import re # regular expression libary.
import nltk

import nltk # Natural Language toolkit
from nltk import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import re
import string
from nltk.corpus import stopwords

import gensim
from gensim import corpora

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

import spacy
import en_core_web_sm


# database connection
database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'
                                               .format("root", "salasana","localhost", "testEmpath"))



################################# Do not remove the above #################################


filename = 'reviews.csv'
df = pd.read_csv(filename)


def EmpathCalculation(df):
    lexicon = Empath()
    df = pd.DataFrame(df)
    Reviews = df["Review"].to_list()
    empathRes = [0] * len(Reviews)
    for ind in range(len(Reviews)):
        empathRes[ind] = lexicon.analyze(Reviews[ind], normalize=False)
        data_items = empathRes[ind].items()
        data_list = list(data_items)

        allCategories = pd.DataFrame(data_list, columns=['Category', 'Score'])
        for Category, Score in allCategories.iteritems():
            '{Category}: {Score}'.format(Category=Category, Score=Score)
        allCategories.head()
        #print(allCategories['Category'])
        allCategories['Category'].to_csv('out2.csv',index=False)
        score = allCategories["Score"]
        score.head()

        nonZero = allCategories[allCategories["Score"] != 0.0]
        nonZero.head()
        nonZero.to_sql(con=database_connection, name='EmpathCategory', if_exists='append', index=False)

def FourcolumnDF():
    # reading nonZero categories from MySQL
    DBSQL = pd.read_sql('SELECT Category, Score FROM EmpathCategory', database_connection)
    DBSQL.head()

    # applying sentiment_cal function to each category with textBlob
    SentCal = DBSQL['Category'].apply(sentiment_calc)

    # Opening the tuple of Sentiment into Polarity and Subjectivity columns
    data_list = list(SentCal)
    allSentiment = pd.DataFrame(data_list, columns=['Polarity', 'Subjectivity'])

    DBSQL['Polarity'] = allSentiment['Polarity']
    DBSQL['Subjectivity'] = allSentiment['Subjectivity']
    #print(DBSQL)

    # checking for non zero sentiment. Can be checked with subjectivity too
    nonZeroSenti = DBSQL[DBSQL["Polarity"] > 0.0]
    nonZeroSenti.to_sql(con=database_connection, name='PositiveSentiment', if_exists='append', index=False)

    nonZeroSenti = DBSQL[DBSQL["Polarity"] < 0.0]
    nonZeroSenti.to_sql(con=database_connection, name='NegativeSentiment', if_exists='append', index=False)

    nonZeroSenti = DBSQL[DBSQL["Polarity"] != 0.0]
    nonZeroSenti.to_sql(con=database_connection, name='Sentiment', if_exists='append', index=False)

def CategorySenti():
    DBSQL = pd.read_sql('SELECT Category, Score FROM EmpathCategory', database_connection)
    DBSQLP = pd.read_sql('SELECT Category, Score FROM EmpathCategory', database_connection)
    #DBSQL.head()
    NegativePol = DBSQL[DBSQL['Category'].isin(
        ["alcohol", "anger", "animal", "anonymity", "breaking", "childish", "cleaning",
         "cold", "confusion", "crime", "death", "deception", "disappointment", "dispute", "domestic_work",
         "envy", "fear", "fire", "hate", "injury", "irritability",
         "kill", "medical_emergency", "monster", "negative_emotion", "neglect", "nervousness", "noise", "pain",
         "poor", "rage", "ridicule", "sadness", "shame", "smell", "stealing", "suffering", "swearing_terms",
         "timidity", "torment", "ugliness", "violence", "weakness", "weapon", "liquid", "rural","sound"
         "law", "disgust","hygiene","fight"])]
    NegativePol.to_sql(con=database_connection, name='NegativePolarity', if_exists='append', index=False)

    PositivePol = DBSQLP[DBSQLP['Category'].isin(
        ["affection","ancient","art","attractive","beach","beauty","celebration","cheerfulness",
         "contentment","dance","divine","economics","emotional","exotic",
         "fashion","fun","gain","hearing","hiking","hipster","independence",
         "joy","leisure","listen","love","medieval","music","ocean",
         "optimism","party","plant","play","politeness","positive_emotion","power","pride",
         "royalty","sexual","sports","surprise","sympathy",
         "toy","trust","urban","valuable","warmth","youth","zest"])]
    PositivePol.to_sql(con=database_connection, name='PositivePolarity', if_exists='append', index=False)

    NeutralPol = DBSQLP[DBSQLP['Category'].isin(
        ["prison","family","furniture","payment","sleep","air_travel","appearance","business","car","cooking",
         "technology","driving","eating","fabric","internet","money","morning","night","office","order","pet","phone",
         "restaurant","shape_and_size","swimming","home"])]
    NeutralPol.to_sql(con=database_connection, name='NeutralPolarity', if_exists='append', index=False)

def sentiment_calc(Category):
    try:
        return TextBlob(Category).sentiment
    except:
        return None

def plotGraphs():
    # Histogram for Positive Sentiment Categories polarised  by TextBlob
    DBSQL = pd.read_sql('SELECT Category FROM PositiveSentiment ', database_connection)
    ax = sns.countplot(y="Category", data=DBSQL)
    ax.set(xlabel="Frequency", ylabel="Categories", title="Frequency of Categories for Positive Sentiment (TextBlob)")
    plt.show()

    # Histogram for Negative Sentiment Categories polarised  by TextBlob
    DBSQL = pd.read_sql('SELECT Category FROM NegativeSentiment ', database_connection)
    ax = sns.countplot(y="Category", data=DBSQL)
    ax.set(xlabel="Frequency", ylabel="Categories", title="Frequency of Categories for Negative Sentiment (TextBlob)")
    plt.show()

    # Bar graph for Category and Polarity polarised  by TextBlob
    DBSQL = pd.read_sql('SELECT Category, Polarity FROM Sentiment ', database_connection)
    bx = sns.barplot(x="Polarity", y="Category", data=DBSQL)
    bx.set(xlabel="Polarity", ylabel="Categories", title="Polarity of Categories (TextBlob)")
    plt.show()

    # Histogram for Negative Sentiment Categories
    DBSQL = pd.read_sql('SELECT Category FROM NegativePolarity ', database_connection)
    ax = sns.countplot(y="Category", data=DBSQL)
    ax.set(xlabel="Frequency", ylabel="Categories", title="Frequency of Categories for Negative Sentiment (Intutively)")
    plt.show()

    # Histogram for Positive Sentiment Categories
    DBSQL = pd.read_sql('SELECT Category FROM PositivePolarity ', database_connection)
    ax = sns.countplot(y="Category", data=DBSQL)
    ax.set(xlabel="Frequency", ylabel="Categories", title="Frequency of Categories for Positive Sentiment (Intutively)")
    plt.show()


    # Histogram for Neutral Sentiment Categories
    DBSQL = pd.read_sql('SELECT Category FROM NeutralPolarity ', database_connection)
    ax = sns.countplot(y="Category", data=DBSQL)
    ax.set(xlabel="Frequency", ylabel="Categories", title="Frequency of Top Categories (Intutively)")
    plt.show()

    #mpld3.fig_to_html(plt, d3_url=None, mpld3_url=None, no_extras=False, template_type='general', figid=None,
     #                 use_http=False)

# Empath Category Calculation
EmpathCalculation(df)

# Sentiment calculation intutively
CategorySenti()

# Sentiment calculation with text blob
FourcolumnDF()

#ploting empath graphs
plotGraphs()
