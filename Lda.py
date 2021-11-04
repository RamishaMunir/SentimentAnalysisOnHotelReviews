filename = 'reviews.csv'
df = pd.read_csv(filename)
def clean_text(text):
    delete_dict = {sp_character: '' for sp_character in string.punctuation}
    delete_dict[' '] = ' '
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    # print('cleaned:'+text1)
    textArr = text1.split()
    text2 = ' '.join([w for w in textArr if (not w.isdigit() and (not w.isdigit() and len(w) > 3))])

    return text2.lower()

df.info()
df.isna().sum()
ax=sns.countplot(x='Rating', data=df)
labels= ['1', '2', '3', '4','5']
colors=['Red', 'Blue', 'Green', 'Orange', 'Black']
sizes= [1355, 1667, 2068, 5739, 8760]
plt.pie(sizes,labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')
plt.axis('equal')
plt.title("Rating Distribution in Percentage")
df.dropna(axis = 0, how ='any',inplace=True)
df['Review'] = df['Review'].apply(clean_text)
df['Num_words_text'] = df['Review'].apply(lambda x:len(str(x).split()))
max_review_data_sentence_length  = df['Num_words_text'].max()
stop_words = stopwords.words('english')

def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text

df['Review']=df['Review'].apply(remove_stopwords)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ']):
    output = []
    for sent in texts:
        doc = nlp(sent)
        output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return output

text_list=df['Review'].tolist()
tokenized_reviews = lemmatization(text_list)
#print("TR",tokenized_reviews)

def lda_modelling():
    # Create vocabulary dictionary and document term matrix
    dictionary = corpora.Dictionary(tokenized_reviews)
    doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]

    # Creating the object for LDA model using gensim library
    LDA = gensim.models.ldamodel.LdaModel

    # Build LDA model
    lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, passes=50)
    lda_model.print_topics()

    for idx, topic in lda_model.show_topics(num_topics=5, num_words=6):
        print('Topic: {} \nWords: {}'.format(idx, topic))

    visualisation = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)
    pyLDAvis.save_html(visualisation, 'LDAplot.html')

#lda model construction
lda_modelling()