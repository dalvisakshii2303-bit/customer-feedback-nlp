# Customer Feedback Sentiment Analysis using NLP

An end-to-end Natural Language Processing and Machine Learning project for analyzing customer reviews, predicting sentiment, and identifying common customer complaint themes.

The project uses TF-IDF text representation and a class-balanced Linear Support Vector Machine (SVM) to classify customer reviews into:

- Positive
- Neutral
- Negative

## 🚀 Live Demo

Try the deployed Streamlit application:

**[Open Customer Feedback Sentiment Analyzer](https://sakshi-customer-feedback-nlp.streamlit.app)**

---

## Project Objective

The objective of this project is to analyze customer feedback and convert unstructured review text into useful business insights.

The project focuses on:

- Customer sentiment classification
- NLP text preprocessing
- Handling class imbalance
- Comparing multiple machine learning models
- Identifying common complaint themes
- Analyzing negative feedback across product departments
- Building an interactive sentiment prediction application

---

## Dataset

The project uses the **Women's Clothing E-Commerce Reviews** dataset.

The original dataset contains:

- **23,486 customer reviews**
- Customer age
- Review title
- Review text
- Product rating
- Recommendation indicator
- Positive feedback count
- Division
- Department
- Product class

---

### Sentiment Distribution

| Sentiment | Percentage |
|---|---:|
| Positive | 77.06% |
| Neutral | 12.47% |
| Negative | 10.47% |

The dataset is highly imbalanced toward Positive reviews, so class-level metrics and Macro F1 were considered alongside overall accuracy.

---

## NLP Pipeline

The text processing pipeline includes:

1. Text normalization
2. Lowercasing
3. Removal of unwanted characters
4. Stopword removal
5. Preservation of important negation words such as `not`
6. Lemmatization
7. TF-IDF vectorization
8. Unigram and bigram feature extraction
9. Sentiment classification

TF-IDF was configured with a maximum of **10,000 features** and:

`ngram_range=(1, 2)`

This allows the model to learn both individual words and two-word phrases such as `not recommend`.

---

## Machine Learning Models

Five model configurations were evaluated.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 82.64% | 0.58 |
| Balanced Logistic Regression | 78.40% | 0.61 |
| Multinomial Naive Bayes | 78.71% | 0.39 |
| Linear SVM | 81.85% | 0.59 |
| **Balanced Linear SVM** | **80.76%** | **0.61** |

Although standard Logistic Regression achieved the highest overall accuracy, the **Balanced Linear SVM** was selected as the final model because it provided a better trade-off between overall accuracy and performance across the imbalanced sentiment classes.

---

## Final Model Performance

### Balanced Linear SVM

**Accuracy:** 80.76%

**Macro F1-score:** 0.61

| Sentiment | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.52 | 0.56 | 0.54 |
| Neutral | 0.38 | 0.36 | 0.37 |
| Positive | 0.92 | 0.91 | 0.91 |

Neutral sentiment remained the most difficult class to identify because neutral reviews often contain a mixture of positive and negative language.

---

## Model Interpretation

Important features learned by the **Balanced Linear SVM** included:
    
### Negative

- awful
- disappointed
- horrible
- not recommend
- unflattering
- worst
- unwearable
- poor

### Positive

- perfect
- love
- comfortable
- great
- happy
- flattering
- pleased
- compliment

Neutral reviews frequently contained contextual terms such as `however`, which reflects the mixed nature of many neutral customer opinions.

---

## Customer Complaint Analysis

Negative reviews were further analyzed using keyword-based issue-theme detection.

| Issue Theme | Reviews Mentioning Theme |
|---|---:|
| Style & Appearance | 1,613 |
| Fit & Size | 1,547 |
| Quality & Material | 1,457 |
| Return Issues | 995 |
| Comfort | 255 |

A review may contain multiple issue themes, so these categories are not mutually exclusive.

The analysis indicates that **Style & Appearance, Fit & Size, and Quality & Material** are among the most frequently detected themes in negative customer feedback.

---

## Department-Level Insights

Observed negative review rates:

| Department | Negative Review Rate |
|---|---:|
| Trend | 17.80% |
| Dresses | 11.08% |
| Tops | 10.91% |
| Jackets | 10.78% |
| Intimate | 8.89% |
| Bottoms | 8.66% |

The Trend department showed the highest observed negative-review rate. However, it contained only 118 reviews, so this result should be interpreted cautiously because of the relatively small sample size.

Additional analysis showed:

- **Bottoms:** Fit & Size represented a particularly large share of detected issue mentions.
- **Tops and Dresses:** Style & Appearance was prominent.
- **Intimate:** Quality & Material was an important issue theme.
- **Jackets:** Return-related issues represented a relatively larger share of detected issues.

---

## Streamlit Application

The project includes an interactive Streamlit application.

Users can enter a customer review and receive a prediction of:

**Positive / Neutral / Negative**

Application workflow:

Customer Review  
↓  
Text Cleaning & Lemmatization  
↓  
TF-IDF Vectorization  
↓  
Balanced Linear SVM  
↓  
Sentiment Prediction

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVM
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

---

## Project Structure

```text
Customer_Feedback_NLP/
│
├── Customer_Feedback_NLP.ipynb
├── app.py
├── sentiment_svm_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Key Learning Outcomes

Through this project, I gained practical experience with:

- NLP preprocessing
- TF-IDF feature engineering
- Text classification
- Imbalanced classification problems
- Precision, recall and F1-score
- Model comparison and selection
- Model interpretation
- Customer complaint analysis
- Business insight generation
- Model serialization
- Building an ML application with Streamlit

---

## Limitations
- The dataset is strongly imbalanced toward Positive reviews.
- Neutral sentiment is more difficult for the model to distinguish.
- Complaint themes are identified using keyword-based rules and should be interpreted as detected issue mentions rather than exact customer complaint labels.
- The model is trained specifically on women's clothing e-commerce reviews, so performance may differ for reviews from other domains.

---

## Future Improvements

Future versions of the project could explore:

- Hyperparameter tuning
- Advanced text representations
- Transformer-based models such as BERT
- Topic modeling
- More advanced complaint classification
- Model explainability techniques
- Deployment for online access

---

## Author

**Sakshi Dalvi**

Statistics | Data Analytics | Python | SQL | Power BI | R | Data Visualization | Machine Learning | NLP
