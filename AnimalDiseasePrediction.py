# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset
df = pd.read_csv("cleaned_animal_disease_prediction.csv")

# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Identify target column
target_col = [col for col in df_encoded.columns if 'Disease_Prediction' in col][0]
X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

# Remove classes with only 1 sample
value_counts = y.value_counts()
valid_classes = value_counts[value_counts > 1].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection to reduce variance (prevent overfitting)
selector = SelectKBest(score_func=f_classif, k=15)
X_selected = selector.fit_transform(X_scaled, y)

# Train-Test Split (Stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

# Regularized Models
models = {
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [100],
        "max_depth": [8, 10],
        "min_samples_split": [4, 6]
    }),
    "Decision Tree": (DecisionTreeClassifier(), {
        "criterion": ["gini"],
        "max_depth": [6, 8],
        "min_samples_split": [4, 6]
    }),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [3, 5],
        "weights": ['uniform']
    })
}

# StratifiedKFold for fair evaluation
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train and evaluate models
model_performance = {}
conf_matrices = {}

for name, (model, params) in models.items():
    grid = GridSearchCV(model, param_grid=params, cv=cv_strategy, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    model_performance[name] = {
        "Best Params": grid.best_params_,
        "Accuracy": round(acc * 100, 2),
        "Classification Report": report
    }
    conf_matrices[name] = confusion_matrix(y_test, preds)

# Naive Bayes (no tuning, inherently low variance)
nb = GaussianNB()
nb.fit(X_train, y_train)
preds_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, preds_nb)
report_nb = classification_report(y_test, preds_nb, output_dict=True)

model_performance["Naive Bayes"] = {
    "Best Params": "Default",
    "Accuracy": round(acc_nb * 100, 2),
    "Classification Report": report_nb
}
conf_matrices["Naive Bayes"] = confusion_matrix(y_test, preds_nb)

# Summary Table
summary_df = pd.DataFrame([{
    "Model": name,
    "Accuracy": model_performance[name]["Accuracy"],
    "Precision": round(model_performance[name]["Classification Report"]["weighted avg"]["precision"], 2),
    "Recall": round(model_performance[name]["Classification Report"]["weighted avg"]["recall"], 2),
    "F1-Score": round(model_performance[name]["Classification Report"]["weighted avg"]["f1-score"], 2)
} for name in model_performance])


print(summary_df)
#Rescale for plotting
summary_df_scaled = summary_df.copy()
summary_df_scaled["Precision"] *= 100
summary_df_scaled["Recall"] *= 100
summary_df_scaled["F1-Score"] *= 100
summary_df_scaled_sorted = summary_df_scaled.sort_values(by="Model").reset_index(drop=True)

# Separate Graphs for Each Metric
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["blue", "orange", "green", "red"]

for metric, color in zip(metrics, colors):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df_scaled_sorted["Model"], summary_df_scaled_sorted[metric], color=color, edgecolor='black')
    plt.title(f'{metric} Comparison of ML Models')
    plt.xlabel('Model')
    plt.ylabel(f'{metric} (%)')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
