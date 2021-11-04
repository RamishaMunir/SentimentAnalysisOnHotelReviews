# %%

import pandas as pd
import re

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import textstat

# %%

data = pd.read_csv("/content/drive/MyDrive/tripadvisor_hotel_reviews.csv")

# %%

data['Polarity_Rating'] = data['Rating'].apply(lambda x: 'Positive' if x > 3 else ('Neutral' if x == 3 else 'Negative'))

# %%

data.to_csv("tripadvisor_hotel_reviews_polarity.csv", index=False)

# %%

dwp = pd.read_csv("tripadvisor_hotel_reviews_polarity.csv")

# %%

# Getting all the reviews termed negative in a single string and forming a word cloud of the string
text1 = ''

for i in dwp[dwp['Polarity_Rating'] == 'Negative']['Review'].values:
    text1 += i + ' '

wc = WordCloud(width=800, height=800, background_color="white", min_font_size=10,
               repeat=True, )
wc.generate(text1)
plt.figure(figsize=(8, 8), facecolor=None)
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.title('Negative' + ' Reviews', fontsize=32)

# %%

# Getting all the reviews termed positive in a single string and forming a word cloud of the string
text2 = ''

for i in dwp[dwp['Polarity_Rating'] == 'Positive']['Review'].values:
    text2 += i + ' '

wc = WordCloud(width=800, height=800, background_color="white", min_font_size=10,
               repeat=True, )
wc.generate(text2)
plt.figure(figsize=(8, 8), facecolor=None)
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.title('Positive' + ' Reviews', fontsize=32)

# %%

# Getting all the reviews termed neutral in a single string and forming a word cloud of the string
text3 = ''

for i in dwp[dwp['Polarity_Rating'] == 'Neutral']['Review'].values:
    text3 += i + ' '

wc = WordCloud(width=800, height=800, background_color="white", min_font_size=10,
               repeat=True, )
wc.generate(text3)
plt.figure(figsize=(8, 8), facecolor=None)
plt.axis("off")
plt.imshow(wc, interpolation="bilinear")
plt.title('Neutral' + ' Reviews', fontsize=32)

# %%

# From the preceding plots we can see that people are mostly talking about their rooms and their day in the hotel.

# %%

# Beach is always a highlight in positive and neutral reviews, while pool and staff are always a highlight in negative reviews.

# %%

# Positive feedbacks are mostly talking about time.

# %%

dwp['Automated_Readability_Index'] = dwp['Review'].apply(textstat.automated_readability_index)

# %%

dwp.to_csv("tripadvisor_hotel_reviews_ari.csv", index=False)

# %%

dwari = pd.read_csv("tripadvisor_hotel_reviews_ari.csv")

# %%

dwari[dwari['Automated_Readability_Index'] < 1].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 2].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 3].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 4].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 5].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 6].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 7].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 8].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 9].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 10].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 11].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 12].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 13].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 14].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

dwari[dwari['Automated_Readability_Index'] < 15].groupby('Polarity_Rating')['Automated_Readability_Index'].count()

# %%

# From the preceding results, we can see that the readability index doesn't refer to Neutral or ambiguous class whatsoever since the results are showing normal distribution.

# %%

# It is perceived that this is due to large sentences.

# %%

searchfor = ['Because of', 'For the purpose of', 'Given that', 'Granted that', 'In fact', 'In order to', 'In view of',
             'Owing to', 'Provided that', 'Seeing that', 'So that', 'With this in mind', 'With this intention',
             'With this purpose']
dwari['Review'].str.contains('|'.join(searchfor), flags=re.IGNORECASE, regex=True)

# %%

searchfor = ['Because of', 'For the purpose of', 'Given that', 'Granted that', 'In fact', 'In order to', 'In view of',
             'Owing to', 'Provided that', 'Seeing that', 'So that', 'With this in mind', 'With this intention',
             'With this purpose']
searchfor_upper = [x.upper() for x in searchfor]
dwari['Argumentation_Result'] = dwari['Review'].apply(
    lambda x: 1 if any(item in str(x).upper() for item in searchfor_upper) else 0)

# %%

dwari['Argumentation_Result']

# %%

dwari.to_csv("tripadvisor_hotel_reviews_argandexp.csv", index=False)

# %%

dwari[dwari['Argumentation_Result'] == 0].groupby('Polarity_Rating')['Argumentation_Result'].count().reset_index(
    name="False_Count")

# %%

dwari[dwari['Argumentation_Result'] == 1].groupby('Polarity_Rating')['Argumentation_Result'].count().reset_index(
    name="True_Count")

# %%

# Even though the number of results is low due to the limited words that were used in the list, the results are showing that argumentation words are usually used more with positive reviews

# %%

searchfor = ['now', 'then', 'soon', 'never', 'forever', 'here', 'there', 'everywhere', 'thence', 'so', 'too', 'nearly',
             'almost', 'quite', 'somewhat', 'therefore', 'consequently', 'purposely', 'wherefore', 'busily',
             'anxiously', 'cleverly']
searchfor_upper = [x.upper() for x in searchfor]
dwari['Explanatory_Result'] = dwari['Review'].apply(
    lambda x: 1 if any(item in str(x).upper() for item in searchfor_upper) else 0)

# %%

dwari['Explanatory_Result']

# %%

dwari.to_csv("tripadvisor_hotel_reviews_argandexp.csv", index=False)

# %%

dwari[dwari['Explanatory_Result'] == 0].groupby('Polarity_Rating')['Explanatory_Result'].count().reset_index(
    name="False_Count")

# %%

dwari[dwari['Explanatory_Result'] == 1].groupby('Polarity_Rating')['Explanatory_Result'].count().reset_index(
    name="True_Count")

# %%

# The number of results is high for explanatory words, which means they are used more frequently, the results are showing that explanatory words are usually used more in both positive and negative, still not as much as positive, but generally more than argumentation words.
