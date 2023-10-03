# Marketplace-price-prediction
At a large scale, determining product pricing becomes increasingly challenging due to the vast number of products sold online. Different product categories, such as clothing and electronics, exhibit unique pricing trends. Clothing prices are heavily influenced by seasonal trends and brand names, while electronics prices fluctuate based on product specifications.

Mercari, Japan's largest community-powered shopping app, is well aware of this complexity. They aim to provide pricing suggestions to sellers on their marketplace. However, this task is difficult as Mercari's sellers have the freedom to list almost anything or any combination of items, making it a complex and diverse marketplace.

Based on the product listing dataset provided by Mercari, I worked on two different projects.
- **Project #1**: Build a predictive tool to estimate the product listing price using regression modeling
- **Project #2**: Perform Topic modeling to identify 10 frequently appeared topics on the description section 
  
GOAL OF the PROJECTS: 
- Use Regression models to predict reasonable Selling Prices based on features of categories on Mercari, an online marketplace.
- Identify clusters of frequently available topics in the text data of product description section by rigorous data cleaning and preprocessing.

## Project 1: Predictive modeling

## Project 2: Topic modeling and clustering
In this project, I performed Topic Modeling to identify 10 frequently appearing topics on the 'item_description' column of the Mercari dataset. I applied text preprocessing involving 

Before starting the MLP process, I visualized the number of the words in 'item_description' dataset:

<img src="images/dist_word_count.png?" width="600" height="300"/>
It seems that the distribution is skewed and the length up to 20 words appears most frequently in the section.

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

<img src="images/cleaned_texts.png?" width="600" height="300"/>

Here, you can identify frequently available words:
<img src="images/top_20_words.png?" width="600" height="300"/>
