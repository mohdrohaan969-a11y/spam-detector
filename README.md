# SMS Spam Detector

A machine learning app that classifies SMS messages as Spam or Not Spam (Ham) using NLP and a Naive Bayes classifier. Built for the AIC2022 course assignment.

**Live demo:** https://spam_checker(ham means harmless).streamlit.app

## Features
- Classifies a message as Spam or Not Spam
- Shows the spam probability
- Simple web interface built with Streamlit

## Technologies Used
- Python
- Scikit-learn (TF-IDF, Naive Bayes, Logistic Regression)
- Pandas, Matplotlib, Seaborn
- Streamlit, Joblib

## Dataset
SMS Spam Collection (5,572 messages; 5,169 after removing duplicates).

## Results
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (default) | 0.9662 | 0.9898 | 0.7405 | 0.8472 |
| Naive Bayes (alpha=0.1) | 0.9816 | 0.9667 | 0.8855 | 0.9243 |
| Logistic Reg. (default) | 0.9603 | 0.9891 | 0.6947 | 0.8161 |
| Logistic Reg. (balanced) | 0.9758 | 0.9206 | 0.8855 | 0.9027 |

The final model is Naive Bayes with alpha=0.1.

## How to Run
1. Install the libraries: `pip install -r requirements.txt`
2. Start the app: `python -m streamlit run app.py`

## Limitations
Trained on SMS data only, so results on emails or other text may be less reliable.
