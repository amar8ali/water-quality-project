import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_and_evaluate(df):
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    # Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    
    # Feature importance visualization
    importances = rf.feature_importances_
    feat_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=feat_names[indices])
    plt.title('Feature Importances (Random Forest)')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save visualization instead of showing
    os.makedirs('reports/figures', exist_ok=True)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'reports/figures/feature_importance_{timestamp}.png', bbox_inches='tight')
    plt.close()
    
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'report': classification_report(y_test, y_pred_rf)
    }
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'report': classification_report(y_test, y_pred_lr)
    }
    # SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    results['SVM'] = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'report': classification_report(y_test, y_pred_svm)
    }
    return results
