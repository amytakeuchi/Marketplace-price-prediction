# Marketplace-price-prediction
At a large scale, determining product pricing becomes increasingly challenging due to the vast number of products sold online. Different product categories, such as clothing and electronics, exhibit unique pricing trends. Clothing prices are heavily influenced by seasonal trends and brand names, while electronics prices fluctuate based on product specifications.

Mercari, Japan's largest community-powered shopping app, is well aware of this complexity. They aim to provide pricing suggestions to sellers on their marketplace. However, this task is difficult as Mercari's sellers have the freedom to list almost anything or any combination of items, making it a complex and diverse marketplace.

Based on the product listing dataset provided by Mercari, I worked on two different projects.
- **Project #1**: Build a predictive tool to estimate the product listing price using regression modeling
- **Project #2**: Perform Topic modeling to identify 10 frequently appeared topics on the description section 
  
GOAL of the projects: 
- Use Regression models to predict reasonable Selling Prices based on features of categories on Mercari, an online marketplace.
- Identify clusters of frequently available topics in the text data of the product description section by rigorous data cleaning and preprocessing.

## Project 1: Predictive modeling

## Project 2: Topic modeling and clustering
In this project, I performed Topic Modeling to identify 10 frequently appearing topics on the 'item_description' column of the Mercari dataset.  
Topic modeling is a statistical and machine learning technique used in natural language processing (NLP) and text mining to identify and extract the underlying topics or themes within a collection of documents. It helps discover patterns and group similar documents together based on the topics they discuss, without requiring human supervision or predefined categories.

In essence, topic modeling helps answer questions like, "What are the main subjects or topics present in a large set of textual data?" Two popular algorithms for topic modeling are Latent Dirichlet Allocation (LDA), which is going to be used in this project, and Non-Negative Matrix Factorization (NMF). These algorithms analyze the co-occurrence patterns of words within documents to uncover topics and their associated word distributions. Researchers and analysts use topic modeling for tasks such as content summarization, recommendation systems, and understanding the main themes in a corpus of text data.

In this project, I followed the NLP pipeline as seen below:

<img src="images/nlp_pipeline.png?" width="400" height="200"/>


Before starting the NLP process, I visualized the number of the words in 'item_description' dataset:

<img src="images/dist_word_count.png?" width="600" height="300"/>
It seems that the distribution is left skewed and the length up to 20 words appears most frequently in the section.

### Text Preprocessing
Next, I preprocessed the text data to apply the NLP model. Here, I removed special characters and stopwords, tokenized, and lemmatized the text data using the following codes:
```
#Preprocessing
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

# Assuming you have a DataFrame named 'train' with a column 'item_description'
# Replace this with your actual DataFrame
train_description = train['item_description']

def preprocess_text(text):
    if isinstance(text, str):  # Check if the value is a string
        # Remove emojis and unicode text
        text = text.encode('ascii', 'ignore').decode('utf-8')
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize the tokens
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a cleaned text
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    else:
        return ""  # Return an empty string for non-string values
df = pd.DataFrame(train_description)
df['cleaned_description'] = df['item_description'].apply(preprocess_text)

# Display the cleaned descriptions
print(df['cleaned_description'])
```
As a result, I could clean the text and create the tokens as follows:

<img src="images/cleaned_texts.png?" width="300" height="150"/>

Here, you can identify frequently available words:

<img src="images/top_20_words.png?" width="600" height="300"/>

As expected, you can find the words that specify the condition, size, and `free shipping` can be found here. 
When I visualized the word crowds of the 'item_description' by the products in different price ranges, the keyword 'new' did not show up as the most frequent keywords in some price range such as listing price in $1,000 - $1,500. So I decided to leave these frequent words for performing the topic modeling while assessing further stop words to remove is one step that I should go through.

### Feature Engineering
I applied TFDIF to vectorize the data.
```
#Vectorizing
tfidf_vectorizer = TfidfVectorizer(min_df=10,
                             max_features=180000,
                             ngram_range=(1, 2))
description_matrix = tfidf_vectorizer.fit_transform(list(df['cleaned_description']))
print('Headline after vectorization: \n{}'.format(description_matrix[0]))
```
### Modeling
For the Topic Modeling, I applied LDA model to identify the top 10 topics of the `item_description` data.
Latent Dirichlet Allocation (LDA) is a probabilistic generative model commonly used for topic modeling in natural language processing. LDA assumes that documents are mixtures of topics, and topics are mixtures of words. It helps uncover the hidden topics within a collection of documents by analyzing the distribution of words across those topics. LDA is based on the idea that documents are created through a process where topics are selected, and then words are generated based on those topics, resulting in a coherent representation of the main themes present in the text data.
```
#LDA
lda_model = LatentDirichletAllocation(n_components=10,
                                      learning_method='online',
                                      max_iter=20,
                                      random_state=42)
X_topics = lda_model.fit_transform(description_matrix)
n_top_words = 20
topic_summaries = []

topic_word = lda_model.components_  # get the topic words
vocab = tfidf_vectorizer.get_feature_names_out()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))
    print('Topic {}: {}'.format(i, ' | '.join(topic_words)))
```
As a result, I could get the top 10 topics as follows:

<img src="images/10_topics.png?" width="550" height="250"/>

Next, I visualized the topics to have a more visible image of the clusters of the topics using t_SNE and PCA.
Both Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) are dimensionality reduction techniques often used for data visualization:

PCA (Principal Component Analysis): PCA reduces the dimensionality of data by transforming it into a new set of uncorrelated variables called principal components. It preserves as much variance as possible in the data while reducing its dimensionality. PCA is often used for linear dimensionality reduction and helps visualize data in lower-dimensional space while maintaining the overall structure and relationships between data points.

t-SNE (t-Distributed Stochastic Neighbor Embedding): t-SNE is a nonlinear dimensionality reduction technique specifically designed for data visualization. It maps high-dimensional data points to a lower-dimensional space while preserving the similarities between data points. t-SNE is particularly useful for visualizing clusters and identifying patterns in the data. It tends to group similar data points together in the lower-dimensional space, making it a popular choice for exploratory data analysis and clustering visualization.

<img src="images/tsne_topics.png?" width="350" height="300"/>

<img src="images/pca_topics.png?" width="350" height="300"/>

Here, it can be pointed out that t-SNE is more effectively capturing the relationship between different topics than PCA. PCA is a linear dimensionality reduction technique, and it may not always be the best choice for visualizing topic modeling results. Topic modeling often deals with non-linear relationships between words and topics, and PCA may not capture these complex structures well. On the other hand,  t-SNE is a non-linear dimensionality reduction technique that is often more effective at preserving the local relationships between data points. When used for visualizing topic modeling results, t-SNE can help reveal clusters of documents or topics that are closely related. It tends to group similar topics or documents together in the lower-dimensional space, making it a popular choice for exploring and visualizing topic clusters.

By plotting the topics using PCA and TSNE, topic #2, #6, and #9, as well as #5, forms relatively large topic clusters with little overlap with other topics while the remaining topics frequently overlap each other. topic #2, #5, #6, and #9 are talking about the topics as follows:
- Topic #2: emphasized discounts and offers including free shipping and bundles
- Topic #5: conditions and colors/sizes of women's apparel
- Topic #6: specifies the condition of Phones
- Topic #9: primarily talking about home goods

While other topics show smaller clusters or/and overlap with each other, these four topics show distinct clusters. For future reference for pricing by listers, examining the pricing in these four categories of topics can be recommended.

