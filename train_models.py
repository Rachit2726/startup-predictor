import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("startup data.csv")

df['status'] = df['status'].apply(lambda x: 1 if x == 'acquired' else 0)

for col in ['category_code', 'state_code', 'city']:
    df[col] = LabelEncoder().fit_transform(df[col])

df.drop(columns=['Unnamed: 0', 'Unnamed: 6', 'state_code.1', 'object_id'], inplace=True, errors='ignore')

# Drop string-type columns before scaling
X = df.drop('status', axis=1)
X = X.select_dtypes(include=[np.number])  # keep only numeric columns
y = df['status']
# Drop any rows with missing values
X.dropna(inplace=True)
y = y[X.index]  # keep labels in sync
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_score = 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_score:
        best_score = acc
        best_model = model

joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and scaler saved!")
