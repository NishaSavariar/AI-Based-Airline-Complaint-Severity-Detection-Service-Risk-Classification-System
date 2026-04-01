AI-Based Airline Complaint Severity Detection & Service Risk Classification System


Project Overview
This project focuses on building an AI-based Natural Language Processing (NLP) system to analyze airline customer reviews and automatically classify:

1.	Complaint Severity – Low, Medium, High, Critical
2.	Service Risk – Risk / No Risk

The system uses Machine Learning, Deep Learning, and Transformer models to compare performance and deploy the best-performing models using a Streamlit web application.
This project helps airlines identify serious customer complaints, service failures, and operational risks early.
________________________________________
Problem Statement
Airlines receive a large number of customer reviews and complaints across various platforms. These reviews contain valuable information about passenger experience, service quality, operational issues, and customer satisfaction. However, manually analyzing thousands of reviews is time-consuming and inefficient.

The objective of this project is to build an AI-based system that automatically analyzes airline customer reviews and performs:
•	Complaint Severity Classification (Low, Medium, High, Critical)
•	Service Risk Classification (Risk / No Risk)
•	Sentiment Analysis
•	Complaint trend analysis
•	Interactive dashboard for business insights

This system helps airlines identify critical complaints, service failures, and operational risks early and improve customer satisfaction and service quality.
________________________________________
Dataset Description
Dataset: Airline Reviews Dataset
Source: airlinequality.com

Dataset Features
Column	Description
Airline Name	Airline company name
Overall Rating	Overall rating given by customer
Review Title	Title of the review
Review Date	Date of review
Verified	Whether review is verified
Review	Review text
Aircraft	Aircraft type
Type Of Traveller	Traveller type
Seat Type	Seat class
Route	Flight route
Date Flown	Flight date
Seat Comfort	Rating
Cabin Staff Service	Rating
Food & Beverages	Rating
Ground Service	Rating
Inflight Entertainment	Rating
Wifi & Connectivity	Rating
Value For Money	Rating
Recommended	Yes / No
	
Dataset Size: 23,171 reviews
________________________________________
Data Preprocessing Steps
Structured Data Cleaning
•	Converted rating columns to numeric
•	Filled missing rating values using median
•	Filled categorical missing values with "Unknown"
•	Removed duplicate reviews

Text Cleaning
Text preprocessing performed:

•	Convert to lowercase
•	Remove URLs
•	Remove HTML tags
•	Remove special characters
•	Remove extra spaces
•	Combine title + review into one column:
full_review_text = Review_Title + Review
________________________________________
Severity Label Creation
Severity labels were created using rating + complaint keywords.

Severity Rules
Condition	Severity
Cancellation / Refund / Lost baggage	Critical
Delay / Rude staff / Bad service	High
Rating ≤ 4	High
Rating ≤ 7	Medium
Rating > 7	Low

Severity distribution:
Severity	Count
High	13355
Critical	4907
Low	2910
Medium	1879

Severity labels encoded using LabelEncoder.
________________________________________
Risk Flag Creation
Binary classification:
Condition	Risk
Rating ≤ 4	Risk
Recommended = No	Risk
Otherwise	No Risk
________________________________________
Exploratory Data Analysis (EDA)
EDA performed:
•	Severity distribution
•	Risk distribution
•	Review length distribution
•	Review length vs severity
•	Overall rating vs severity
•	Wordcloud for Low & Critical reviews
•	Sentiment analysis using TextBlob
•	Sentiment vs severity
•	Review trends over time
EDA helps understand complaint patterns and customer behavior.
________________________________________
Model Building
Three model families were implemented:

1. Machine Learning Models

Severity Classification
•	Logistic Regression
•	Linear SVM
•	Multinomial Naive Bayes
Risk Classification
•	Logistic Regression
•	Random Forest

Feature Engineering
Used TF-IDF (unigrams + bigrams).

2. Deep Learning Models
Used BiLSTM Neural Networks.
Architecture
Embedding Layer
Bidirectional LSTM
Dropout
Dense Layer
Output Layer

Built two models:
•	Severity Classification (Softmax output)
•	Risk Classification (Sigmoid output)

3. Transformer Models

Used DistilBERT (HuggingFace Transformers).

Tasks:
•	Severity Classification (4 classes)
•	Risk Classification (Binary)
Used:
•	Tokenizer
•	Trainer API
•	Fine-tuning
•	Evaluation

Transformer models provided best performance.
________________________________________
Model Evaluation
Severity Classification Metrics
•	Accuracy
•	Precision
•	Recall
•	F1 Score (Macro)
•	Confusion Matrix
Risk Classification Metrics
•	Accuracy
•	Precision
•	Recall
•	F1 Score
•	ROC-AUC

Model Comparison
Compared:
•	ML vs DL vs Transformer
•	Accuracy
•	F1 Score

Final Model Selection:
Task	Final Model
Severity	DistilBERT
Risk	Random Forest

Reason:
•	Transformer best for text understanding
•	Random Forest performs well for binary classification
•	Faster inference
________________________________________
Results & Insights
Model Performance Summary

Model	Task	Accuracy	F1 Score
Logistic Regression	Severity	~0.76	~0.57
SVM	Severity	~0.77	~0.60
Naive Bayes	Severity	~0.70	~0.50
BiLSTM	Severity	Good	Good
DistilBERT	Severity	Best	Best
Random Forest	Risk	Best	Best
			
Key Insights:

•	Transformer models performed best for text classification.
•	Random Forest performed well for risk classification.
•	Reviews with words like cancel, refund, delay, lost baggage often correspond to Critical severity.
•	Negative sentiment reviews are highly correlated with Risk classification.
•	Longer reviews tend to correspond to higher severity complaints.
•	Most complaints fall under High and Critical severity categories.
________________________________________
Streamlit Application
A multi-page Streamlit dashboard was developed.

Pages:

Page 1 – Introduction
•	Project overview
•	Dataset description
•	Business use cases

Page 2 – EDA Dashboard
•	Severity distribution
•	Risk distribution
•	Review length analysis

Page 3 – Prediction
•	Single review prediction
•	CSV bulk prediction
•	Download predictions

Prediction Output
The system predicts:
•	Complaint Severity
•	Service Risk

Models used:
•	Severity → DistilBERT
•	Risk → Random Forest
________________________________________
Project Architecture
User Review
     ↓
Text Preprocessing
     ↓
Severity Model (DistilBERT)
Risk Model (Random Forest)
     ↓
Prediction Output
     ↓
Streamlit Dashboard
________________________________________
Technologies Used
Programming Language
•	Python
Libraries & Frameworks
•	Pandas – Data processing
•	NumPy – Numerical operations
•	Scikit-learn – Machine learning models
•	TensorFlow / Keras – Deep learning models
•	HuggingFace Transformers – DistilBERT model
•	TextBlob – Sentiment analysis
•	Matplotlib / Seaborn – Data visualization
•	WordCloud – Text visualization
•	Streamlit – Web application
•	Joblib / Pickle – Model saving and loading
Tools
•	Jupyter Notebook
•	VS Code
•	GitHub
Deployment (Optional)
•	AWS EC2
•	AWS S3
•	AWS RDS
________________________________________Setup Instructions
Install Dependencies
pip install pandas
pip install numpy
pip install scikit-learn
pip install tensorflow
pip install torch
pip install transformers
pip install streamlit
pip install joblib
pip install textblob
pip install wordcloud
pip install seaborn
pip install matplotlib

Run Streamlit App
streamlit run app.py
________________________________________
Usage Guide
Single Prediction
1.	Open Streamlit app
2.	Go to Prediction page
3.	Enter airline review text
4.	Click Predict
5.	System shows Severity and Risk

Bulk Prediction
1.	Upload CSV file with column name review
2.	System predicts severity and risk
3.	Download results as CSV
________________________________________
Project Folder Structure
Airline_Final_Project/
│
├── data/
│   └── cleaned_airline_reviews.csv
│   └── Airline_reviews.csv
│
├── models/
│   ├── tfidf.pkl
│   ├── severity_lr.pkl
│   ├── severity_svm.pkl
│   ├── severity_nb.pkl
│   ├── risk_rf.pkl
│   ├── dl_severity_model.keras
│   ├── dl_risk_model.keras
│   ├── tokenizer.pkl
│   ├── distilbert_severity/
│   └── distilbert_risk/
│
├── app.py
├── Data Processing.ipynb
├── requirements.txt
└── README.md
________________________________________


Project Deliverables

The final project includes the following deliverables:

•	Complete source code
•	Cleaned dataset
•	Machine Learning models
•	Deep Learning models
•	Transformer models
•	Model comparison results
•	Streamlit web application
•	Bulk prediction system
•	README documentation
•	requirements.txt file
•	Project presentation (PPT)
•	GitHub repository
________________________________________
Business Use Cases
•	Airline service quality monitoring
•	Customer dissatisfaction detection
•	Risk and reputation management
•	Complaint trend analysis
•	Customer experience analytics
________________________________________
Future Improvements

The following improvements can be implemented in future versions of this project:

•	Deploy the application on AWS EC2 for public access
•	Store models in AWS S3
•	Add explainable AI using SHAP or LIME
•	Add airline comparison dashboard
•	Add real-time review scraping system
•	Add confidence scores for predictions
________________________________________
Conclusion
This project successfully developed an AI-based Airline Complaint Severity Detection and Service Risk Classification System using Machine Learning, Deep Learning, and Transformer models. The system can automatically analyze airline reviews, classify complaint severity, detect service risks, and provide insights through an interactive Streamlit dashboard.
________________________________________
