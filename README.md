# Sentiment Analysis of Customer Reviews
## Project Overview

This project involves performing sentiment analysis on customer reviews using machine learning techniques. The goal is to classify reviews into three sentiment categories: Positive, Neutral, and Negative. The analysis provides insights into customer satisfaction and identifies areas for improvement.

## Dataset

- The dataset contains the following columns:

- Product Name: Name of the product being reviewed.

- Brand Name: Brand associated with the product.

- Price: Price of the product.

- Rating: Numerical rating given by the customer.

- Reviews: Textual reviews provided by customers.

- Review Votes: Number of votes on the review's helpfulness.

## Problem Statement
The task is to:

1) Preprocess the text data to clean and tokenize the reviews.

2) Convert textual data into numerical features using TF-IDF.

3) Train multiple machine learning models to classify the reviews into sentiment categories.

4) Evaluate the models and visualize their performance.

5)Focus on the best-performing model (Random Forest Classifier) for deeper analysis.

## Preprocessing Steps

**1. Cleaning Reviews**
   - Removed URLs, punctuation, and special characters.

   - Converted text to lowercase.

   - Removed stop words.

   - Applied lemmatization to standardize words.
  
**2. Assigning Sentiments:**

   - Ratings ≥ 4: Positive

   - Rating = 3: Neutral

   - Ratings ≤ 2: Negative

## Models Used

**1. Logistic Regression**

**2. Random Forest Classifier**

**3. Decision Tree Classifier**

## Evaluation Metrics

1. **Accuracy**: Overall performance of the models.

2. **Classification Report**: Precision, Recall, and F1-score for each sentiment class.

3. **Confusion Matrix**: Visualized using a heatmap to analyze misclassifications.

## Key Findings

- **Random Forest Classifier** outperformed other models with the highest accuracy.

- Positive sentiments were classified accurately, while some negative and neutral reviews were misclassified.

- Word Clouds were generated for different sentiment labels to visualize common terms.

## Visualization

**Confusion Matrix:** Displayed for the Random Forest Classifier to highlight prediction performance.

**Word Clouds**: Separate Word Clouds for Positive, Neutral, and Negative reviews to identify frequent terms.

## Code Implementation Highlights

1. **Data Splitting**:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


2. **TF-IDF Vectorization**:
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer(max_features=5000)
   X_train_vectorized = vectorizer.fit_transform(X_train)
   X_test_vectorized = vectorizer.transform(X_test)

3. **Model Training:**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train_vectorized, y_train)

4. **Visualization:**:
   - **Confusion Matrix**:
     ```python
     from sklearn.ensemble import RandomForestClassifier
     model = RandomForestClassifier(n_estimators=100, random_state=42)
     model.fit(X_train_vectorized, y_train)

    - **Word Clouds**:
      ```python
      from wordcloud import WordCloud
      wordcloud = WordCloud().generate(" ".join(reviews))
   
## Business Insights

1. **Customer Satisfaction:**

  - Majority of customers are satisfied based on the high proportion of positive reviews.

2. **Actionable Feedback:**

  - Negative reviews reveal common pain points like product quality or delivery issues.

3. **Targeted Engagement:**

  - Neutral reviews offer an opportunity to engage and convert undecided customers into promoters.

## Future Work

1. Use advanced NLP models like LSTM, BERT, or GPT for better accuracy.

2. Compare all the models (traditional and advanced) to identify the best performer.

3. Integrate sentiment insights into a dashboard for real-time monitoring.





