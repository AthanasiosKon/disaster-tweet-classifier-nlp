from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def run_baseline(X_train, X_val, y_train, y_val):
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_val_tfidf)

    print("\nBaseline: TF-IDF + Logistic Regression")
    print(classification_report(y_val, preds))
