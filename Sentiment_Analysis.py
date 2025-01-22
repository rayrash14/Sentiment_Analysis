#!/usr/bin/env python
# coding: utf-8

# ## Step 1: Import Necessary Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Generate separate Word Clouds for each label
from wordcloud import WordCloud


# ## Step 2: Load Dataset

# In[3]:


df = pd.read_csv('/Users/rashmi/Downloads/Amazon_Unlocked_Mobile.csv')
df.head()


# #### Initial Informations related to Dataset

# In[4]:


df.info()


# In[5]:


df['Reviews']


# In[6]:


for review in df['Reviews'].head(20):
    print(review)


# In[7]:


unique_values = df.nunique()
print(unique_values)


# The dataset comprises 413,840 entries distributed across six columns. 
# The Product Name and Rating columns have no missing values, 
# while others, such as Brand Name (with 65,171 missing entries), Price (5,933 missing entries), 
# Reviews (70 missing entries), and Review Votes (12,296 missing entries), contain varying levels of missing data.

# ## Step 3: Data Pre-processing

# In[8]:


df['Reviews'] = df['Reviews'].str.lower()
df['Reviews'].fillna('', inplace=True)
print("\nFirst few rows after preprocessing:")
print(df.head())


# ## Step 4: Label The Dataset

# In[9]:


def LabelFunc(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['Label'] = df['Rating'].apply(LabelFunc)
# Display the first few rows with sentiment
print("\nFirst few rows with sentiment derived from rating:")
print(df.head())


# In[10]:


# Define stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()


# In[11]:


# Define the text preprocessing function
def clean_Review(review_text):
    review_text = re.sub(r"http\S+", "", review_text)  # Remove URLs
    review_text = re.sub(r"[^a-zA-Z]", " ", review_text)  # Remove numbers and punctuation
    review_text = str(review_text).lower()  # Convert to lowercase
    review_text = word_tokenize(review_text)  # Tokenize text
    review_text = [item for item in review_text if item not in stop_words]  # Remove stop words
    review_text = [lemma.lemmatize(word, pos='v') for word in review_text]  # Lemmatize words
    review_text = [i for i in review_text if len(i) > 2]  # Remove short words
    review_text = " ".join(review_text)  # Rejoin words into a single string
    return review_text

# Handle missing reviews
df['Reviews'].fillna('', inplace=True)

# Apply the preprocessing function to the 'Reviews' column
df['CleanReview'] = df['Reviews'].apply(clean_Review)

# Display the original and cleaned reviews
print(df[['Reviews', 'CleanReview']].head())


# ## Step 5: Feature Extraction

# In[12]:


X = df['CleanReview']
y = df['Label']


# ## Step 6: Split the Data

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ## Step 7: Convert Text To Numerical Data

# In[15]:


# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# ## Step 8: Train & Check Accuracy Of Multiple Models
# 

# In[16]:


# Initialize multiple models 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train each model and evaluate performance
results = {}

for model_name, model in models.items():
    # Train the model
    model.fit(X_train_vectorized, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_vectorized)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=y_test.unique())
    
    # Store the results
    results[model_name] = accuracy
    
    # Print accuracy and classification report
    print(f"{model_name}: Accuracy = {accuracy:.2f}")
    print(f"Classification Report for {model_name}:")
    print(report)

# Display the results
print("\nModel Performance:")
for model, acc in results.items():
    print(f"{model}: {acc:.2f}")


# We see that Random Forest Classifier perform really well on the given data.

# ## Step 9: Visualize The Result for Random Forest Classifier model

# In[17]:


# Get predictions for Random Forest
rf_model = models["Random Forest"]
y_pred_rf = rf_model.predict(X_test_vectorized)

conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()


# ## Step 10: Generate wordcloud

# In[18]:


labels = y_test.unique()
for label in labels:
    # Filter reviews by label
    reviews_by_label = df[df['Label'] == label]['CleanReview']
    combined_reviews = " ".join(reviews_by_label)

    # Create the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=200).generate(combined_reviews)

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {label} Reviews')
    plt.show()


# In[ ]:




