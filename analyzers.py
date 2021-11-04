from database import Database
import env
import subprocess
import shlex
import os.path
import sys
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import pearsonr
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn import preprocessing
import time
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import wordnet as wn
# from pandasgui import show
# from pandasgui.datasets import titanic
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

nlp = spacy.load('en_core_web_sm')

def create_db():
    schema_name = env.master_db
    db_conf = env.db_production
    rdb = Database(schema=schema_name, **db_conf)
    engine = rdb.create_engine()
    return engine
connection = create_db()

def RateSentimentRawData(df):
    df_raw_data_updated = df
    df_raw_data_updated['raw_sentiment_type'] = ''
    df_raw_data_updated['Rating'] = (df_raw_data_updated.Rating).astype(int)
    df_raw_data_updated.loc[df_raw_data_updated.Rating > 3, 'raw_sentiment_type'] = 'POSITIVE'
    df_raw_data_updated.loc[df_raw_data_updated.Rating == 3, 'raw_sentiment_type'] = 'NEUTRAL'
    df_raw_data_updated.loc[df_raw_data_updated.Rating < 3, 'raw_sentiment_type'] = 'NEGATIVE'
    return df_raw_data_updated

def RateSentimentUsingSS(df):
    #open a subprocess using shlex to get the command line string into the correct args list format
    df_text = df['Review']
    df_text = df_text.str.replace('\n', '')
    df_text = df_text.str.replace('\r', '')
    conc_text = '\n'.join(df_text)
    cmd = f'java -jar "{env.SentiStrengthLocation}" stdin sentidata "{env.SentiStrengthLanguageFolder}"'
    p = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    b = bytes(conc_text.replace(" ", "+"), 'utf-8')
    stdout_byte, stderr_text = p.communicate(b)
    stdout_text = stdout_byte.decode("utf-8")
    stdout_text = stdout_text.rstrip().replace("\t", " ")
    stdout_text = stdout_text.replace('\r\n', '')
    senti_score = stdout_text.split(' ')
    senti_score = list(map(float, senti_score))
    senti_score = [tuple(senti_score[i:i + 2]) for i in range(0, len(senti_score), 2)]
    senti_analysis = dict(zip(df_text.to_list(), senti_score))
    final_df = pd.DataFrame.from_dict(senti_analysis, orient='index').reset_index().rename(columns={'index': 'Review',
                                                                                        0: 'ss_positive', 1: 'ss_negative'})
    final_df['ss_overall_senti_score'] = final_df['ss_positive'] + final_df['ss_negative']
    final_df['ss_sentiment_type'] = ''
    final_df.loc[final_df.ss_overall_senti_score > 0, 'ss_sentiment_type'] = 'POSITIVE'
    final_df.loc[final_df.ss_overall_senti_score == 0, 'ss_sentiment_type'] = 'NEUTRAL'
    final_df.loc[final_df.ss_overall_senti_score < 0, 'ss_sentiment_type'] = 'NEGATIVE'
    final_df = df.merge(final_df, on='Review', how='left').reset_index()
    return final_df

def RateSentimentUsingNltkVader(df):
    sid = SentimentIntensityAnalyzer()
    df['vader_scores'] = df['Review'].apply(lambda x: sid.polarity_scores(x))
    df['vader_compound'] = df['vader_scores'].apply(lambda score_dict: score_dict['compound'])
    df['vader_negative'] = df['vader_scores'].apply(lambda score_dict: score_dict['neg'])
    df['vader_positive'] = df['vader_scores'].apply(lambda score_dict: score_dict['pos'])
    df['vader_neutral'] = df['vader_scores'].apply(lambda score_dict: score_dict['neu'])
    df['vader_overall_score'] = df['vader_positive'] + df['vader_negative']
    df.drop(['vader_scores'], axis=1, inplace=True)
    df['vader_sentiment_type'] = ''
    df.loc[df.vader_compound > 0, 'vader_sentiment_type'] = 'POSITIVE'
    df.loc[df.vader_compound == 0, 'vader_sentiment_type'] = 'NEUTRAL'
    df.loc[df.vader_compound < 0, 'vader_sentiment_type'] = 'NEGATIVE'
    return df

def count_of_pronouns(row):
    count = 0
    for token in nlp(row):
        if token.pos_ in ('PRON'):
            count = count + 1
    return count


# Function to extract the proper nouns

def ProperNounExtractor(text):
    sentences = nltk.sent_tokenize(text)
    noun_count = 0
    pronoun_count = 0
    noun_list = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word for word in words if word not in set(stopwords.words('english'))]
        tagged = nltk.pos_tag(words)
        for (word, tag) in tagged:
            if tag in ('NNP', 'NNS', 'NNP', 'NNPS'):  # If the word is a proper noun
                noun_count = noun_count + 1
                noun_list.append(word)
            elif tag in ('PRP', 'PRP$', 'WP', 'WP$'):
                pronoun_count = pronoun_count + 1
    return noun_count, pronoun_count

def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            if current_chunk:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                            continuous_chunk.append(named_entity)
                            current_chunk = []
            else:
                    continue
    return continuous_chunk

def extract_tokens_noun_pronoun(df):
    query = "select * From complete_analysis_on_hotel_reviews"
    df = pd.read_sql(query)

    reviews = df['Review']
    # 5. tokenization and nouns length
    start = time.process_time()
    df['tokens_length'] = [len(nltk.word_tokenize(i)) for i in reviews]
    print(time.process_time() - start)
    # a,b = reviews.apply(lambda x: ProperNounExtractor(x))
    # df['tokens_length'] = [len(list(TextBlob(i).words)) for i in reviews]

    start = time.process_time()
    df['nouns_length'] = [len(list(TextBlob(i).noun_phrases)) for i in reviews]
    print(time.process_time() - start)

    df['pronouns_length'] = reviews.apply(lambda x: count_of_pronouns(x))

    # calculate the corelation among each
    # token_len vs analyzer 1 and 2
    corr_token_len_user = df['tokens_length'].corr(df['Rating'], method='pearson', min_periods=None)
    corr_token_len_ss = df['tokens_length'].corr(df['ss_overall_senti_score'], method='pearson', min_periods=None)
    corr_token_len_vader = df['tokens_length'].corr(df['vader_overall_score'], method='pearson', min_periods=None)

    corr_noun_len_user = df['nouns_length'].corr(df['Rating'], method='pearson', min_periods=None)
    corr_noun_len_ss = df['nouns_length'].corr(df['ss_overall_senti_score'], method='pearson', min_periods=None)
    corr_noun_len_vader = df['nouns_length'].corr(df['vader_overall_score'], method='pearson', min_periods=None)

    corr_pronoun_len_user = df['pronouns_length'].corr(df['Rating'], method='pearson', min_periods=None)
    corr_pronoun_len_ss = df['pronouns_length'].corr(df['ss_overall_senti_score'], method='pearson', min_periods=None)
    corr_pronoun_len_vader = df['pronouns_length'].corr(df['vader_overall_score'], method='pearson', min_periods=None)

    print(f"Correlation of Token Length vs User Rating: {corr_token_len_user}\n"
        f"Correlation of Token Length vs SentiStrength: {corr_token_len_ss}\n"
          f"Correlation of Token Length vs VaderNLTK: {corr_token_len_vader}\n"
          f"Correlation of Nouns Length vs User Rating: {corr_noun_len_user}\n"
          f"Correlation of Nouns Length vs SentiStrength: {corr_noun_len_ss}\n"
          f"Correlation of Nouns Length vs VaderNLTK: {corr_noun_len_vader}\n"

          f"Correlation of Pronouns Length vs User Rating: {corr_pronoun_len_user}\n"
          f"Correlation of Pronouns Length vs SentiStrength: {corr_pronoun_len_ss}\n"
          f"Correlation of Pronouns Length vs VaderNLTK: {corr_pronoun_len_vader}"
          )

    print(f"{corr_token_len_ss}{corr_token_len_vader}{corr_noun_len_ss}{corr_noun_len_vader}{corr_pronoun_len_ss}{corr_pronoun_len_vader}")

def count_bad_words(review):
    count = 0
    review = review.replace(',', '').replace('-', ' ').replace('.', ' ').replace('\n', ' ').replace('\t', ' ')
    review_list = review.split(' ')
    for item in review_list:
        if len(wn.synsets(item)) == 0 and item not in ['', None]:
            count = count + 1
    return count

def verify_hypothesis():
    query = "select * From complete_analysis_on_hotel_reviews"
    df = pd.read_sql(query, connection)
    # normalize
    x = df[['ss_overall_senti_score', 'vader_overall_score']].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    x_scaled = min_max_scaler.fit_transform(x)
    df[['ss_overall_senti_score', 'vader_overall_score']] = pd.DataFrame(x_scaled.round())
    df['diff_ss_user'] = (df['ss_overall_senti_score'] - df['Rating']).round()

    # determine ambigous class
    df['ambigous_class'] = 'No'
    df.loc[(df['diff_ss_user'] <= -2) | (df['diff_ss_user'] >= 2), 'ambigous_class'] = 'Yes'

    # test hypothesis1: badly written reviews are likely to be included in this ambiguous class
    df['bad_reviews_count'] = df['Review'].apply(lambda x: count_bad_words(x))
    df_hyp1 = df.groupby(['ambigous_class'])['bad_reviews_count'].sum().reset_index()

    # test hypothesis2: ambiguous reviews are likely to be short
    df['tokens_length'] = [len(nltk.word_tokenize(i)) for i in df['Review']]
    df_hyp2 = df.groupby(['ambigous_class'])['tokens_length'].sum().reset_index()

    for item, values in df_hyp1.iterrows():
        print(f"Ambigous Class: {item}, Bad Review Count: {values}")
    for item, values in df_hyp2.iterrows():
        print(f"Ambigous Class: {item}, Tokens Length: {values}")

    return df_hyp1, df_hyp2


#used in UI
fig = None
def normalize_and_plot():
    query = "select * From complete_analysis_on_hotel_reviews"
    df = pd.read_sql(query, connection)
    # normalize
    x = df[['ss_overall_senti_score', 'vader_overall_score']].values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 5))
    x_scaled = min_max_scaler.fit_transform(x)
    df[['ss_overall_senti_score', 'vader_overall_score']] = pd.DataFrame(x_scaled)

    df_plot = df[['idx', 'Rating', 'ss_overall_senti_score', 'vader_overall_score']]
    new_df = pd.DataFrame({'idx': [1, 2, 3, 4, 5]})
    for i in new_df['idx']:
        a = df_plot[df_plot['Rating'].astype(int) == int(i)].shape[0]
        b = df_plot[df_plot['ss_overall_senti_score'].astype(int) == int(i)].shape[0]
        c = df_plot[df_plot['vader_overall_score'].round() == int(i)].shape[0]
        new_df.loc[i - 1, 'Rating'] = int(a)
        new_df.loc[i - 1, 'ss_overall_senti_score'] = int(b)
        new_df.loc[i - 1, 'vader_overall_score'] = int(c)

    fig = new_df.rename(columns={'idx': 'Polarity of Reviews','Rating': 'User Score', 'ss_overall_senti_score': 'SentiStrength Score',
                           'vader_overall_score': 'vaderNLTK Score'}).plot(x='Polarity of Reviews', y=['User Score', 'SentiStrength Score', 'vaderNLTK Score'], kind="bar")
    fig = fig.get_figure()
    plt.show()

def get_ss_rating(idx):
    query = f"select ss_positive, ss_negative, ss_overall_senti_score, review from complete_analysis_on_hotel_reviews " \
            f"where idx = {idx}"
    results = connection.execute(query)
    return results.first()

def get_vader_rating(idx):
    query = f"select vader_positive, vader_negative, vader_overall_score, review from complete_analysis_on_hotel_reviews " \
            f"where idx = {idx}"
    results = connection.execute(query)
    return results.first()

def calculate_correlation():
    query_complete = "select * From complete_analysis_on_hotel_reviews"
    final_df = pd.read_sql(query_complete, connection)
    print("================================================")
    corr_raw_ss = final_df['ss_overall_senti_score'].corr(final_df['Rating'], method='pearson', min_periods=None)
    corr_raw_vader = final_df['vader_overall_score'].corr(final_df['Rating'], method='pearson', min_periods=None)
    corr_ss_vader = final_df['vader_overall_score'].corr(final_df['ss_overall_senti_score'], method='pearson', min_periods=None)

    print(f"Correlation of User Rating vs SentiStrength: {corr_raw_ss}\n"
          f"Correlation of User Rating vs VaderNLTK: {corr_raw_vader}\n"
          f"Correlation of SentiStrength vs VaderNLTK: {corr_ss_vader}")

    # point 4:
    cond_pos = final_df['raw_sentiment_type'] == 'POSITIVE'
    cond_neg = final_df['raw_sentiment_type'] == 'NEGATIVE'
    set1 = final_df.loc[cond_pos, 'Rating']
    set2 = final_df.loc[cond_neg, 'Rating']

    corr_raw_pos_vader = final_df.loc[cond_pos, 'vader_overall_score'].corr(set1,
                                                                            method='pearson', min_periods=None)
    corr_raw_neg_vader = final_df.loc[cond_neg, 'vader_overall_score'].corr(set2,
                                                                            method='pearson', min_periods=None)
    corr_raw_neg_ss = final_df.loc[cond_neg, 'ss_overall_senti_score'].corr(set2,
                                                                            method='pearson', min_periods=None)
    corr_raw_pos_ss = final_df.loc[cond_pos, 'ss_overall_senti_score'].corr(set1,
                                                                            method='pearson', min_periods=None)

    print(
        f"Correlation of Positive User Rating vs SentiStrength: {corr_raw_pos_ss}\n"
        f"Correlation of Negative User Rating vs SentiStrength: {corr_raw_neg_ss}\n"
        f"Correlation of Positive User Rating vs VaderNLTK: {corr_raw_pos_vader}\n"
        f"Correlation of Negative User Rating vs VaderNLTK: {corr_raw_neg_vader}")
    return corr_raw_ss, corr_raw_vader, corr_ss_vader, corr_raw_pos_ss, corr_raw_neg_ss, corr_raw_pos_vader, corr_raw_neg_vader



if __name__ == '__main__':
    query = "select * From tripadvisor_hotel_reviews"
    df = pd.read_sql(query, connection)

    #12
    verify_hypothesis()

    print("=========Generating Sentiments On Raw Data=========")
    final_df = RateSentimentRawData(df)
    # final_df.to_sql('tripadvisor_hotel_reviews_updated', connection, if_exists='replace')

    print("=========Starting Analysis with SentiStrength=========")
    final_df = RateSentimentUsingSS(final_df)
    # final_df.to_sql('sentistrength_analysis_on_hotel_reviews', connection, if_exists='replace')

    print("=========Starting Analysis with VaderNLTK=========")
    final_df = RateSentimentUsingNltkVader(final_df)
    # final_df.to_sql('nltk_vader_analysis_on_hotel_reviews', connection, if_exists='replace')
    # final_df.sentiment_type.value_counts().plot(kind='bar', title="sentiment analysis")

    final_df = final_df.rename(columns={"index": "idx"})
    # final_df.to_sql('complete_analysis_on_hotel_reviews', connection, if_exists='replace')

    # 3. Plot
    # normalize the data and plot
    normalize_and_plot()

    #3.person corelation
    calculate_correlation()

    #5:
    extract_tokens_noun_pronoun(df)

    print('Done')
