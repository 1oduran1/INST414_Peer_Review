import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# === Utility Functions ===
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values('Date')  # Ensure chronological order
    df = df.drop(columns=['Region Type', 'State Name', 'Region Name', 'Year', 'Size Rank', 'Monthly Payment'])
    X = df.drop(columns=['Affordability', 'Date'])
    y = df['Affordability']
    return X, y

def evaluate_with_time_series_split(name, model, X, y):
    tscv = TimeSeriesSplit()
    #print(f'=== {name} - Time Series Split Evaluation ===')
    
    scores = []
    all_preds = []
    all_true = []
    fold = 1
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        fold += 1

        all_preds.extend(y_pred)
        all_true.extend(y_test)
    
    average_score = np.mean(scores)
    #return average_score
    print(f'Average accuracy score of {name}: {average_score:.4f}')
    
    #visualize
    cm = confusion_matrix(all_true, all_preds)
    cm_percent = cm / cm.sum(axis=1, keepdims=True)

    labels = [[f"{count}\n({percent:.1%})" for count, percent in zip(row_c, row_p)] 
          for row_c, row_p in zip(cm, cm_percent)]
    # Plot it
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_percent, annot=labels, fmt='', cmap='Blues',
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f'{name} Confusion Matrix (TimeSeriesSplit)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
    


# === Load Data ===
X, y = load_and_prepare_data('/Users/oliviaduran/Desktop/Calc_Median_Incomes_Affordability.csv')
# === Define Models ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=500, random_state=0),
    "XGBoost": XGBClassifier(n_estimators=1300, random_state=0)
}

# === Run Evaluations ===
for name, model in models.items():
    evaluate_with_time_series_split(name, model, X, y)
    
'''# === Fine Tuning ===
results = {}
#fine tuning B for both models
for n in np.arange(100, 1000, 100):
    for learning_rate in list(0.001, 0.01, 0.1, 1):
        for d in np.arange(0, )
        # Update model with new n_estimators
        setattr(models["XGBoost"], 'n_estimators', n)
        
        # Evaluate
        score = evaluate_with_time_series_split("XGBoost", models["XGBoost"], X, y)
        
        # Store results
        results[(n, lambda)] = score
        print(f"XGBoost, n={n}, lambda={lambda}: mean CV accuracy = {score:.4f}")
    for model_name in models:
        # Update model with new n_estimators
        setattr(models[model_name], 'n_estimators', n)
        
        # Evaluate
        score = evaluate_with_time_series_split(model_name, models[model_name], X, y)
        
        # Store results
        print(f"{model_name}, n={n}: mean CV accuracy = {score:.4f}")'''
