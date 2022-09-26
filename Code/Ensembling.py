import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

datafile = 'results/inference_23Sept.xlsx'
#datafile = 'results/validation_62.xlsx'

df = pd.read_excel (datafile)

# Convert Labels from string to int - Create mapping
label = {'SUPPORTS': 1, 'REFUTES': 0, 'NO MATCHING ARTICLES FOUND': 2}

df['ground_truth'] = df['ground_truth'].map(label)
df['rev_img'] = df['rev_img'].map(label)
df['vis_bert'] = df['vis_bert'].map(label)
df['text_cls'] = df['text_cls'].map(label)

# Remove NaN
df = df.fillna(0)

df['rev_img'] = df['rev_img'].astype(int)
df['vis_bert'] = df['vis_bert'].astype(int)
df['text_cls'] = df['text_cls'].astype(int)
df['ground_truth'] = df['ground_truth'].astype(int)

X = df.iloc[:, 2:5].values
y = df.iloc[:, 6].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training
Forest_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
Forest_classifier.fit(X_train, y_train)
y_pred = Forest_classifier.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred, digits = 4))
print(accuracy_score(y_test, y_pred))

