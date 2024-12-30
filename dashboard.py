import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import threading
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from pyngrok import ngrok

# Streamlit setup
st.title("K-Means Clustering & Logistic Regression")
st.write("Visualisasi hasil analisis menggunakan algoritma K-Means dan Logistic Regression.")

# Load dataset
@st.cache_data
def load_data():
    file_path = "Customer_Segmentation.csv"  # Replace with your file path
    return pd.read_csv(file_path)

data = load_data()

# Sidebar for data exploration
st.sidebar.title("Data Exploration")
if st.sidebar.checkbox("Show Raw Data"):
    st.write("### Raw Dataset", data)

# Step 1: Data Preparation
st.write("## Step 1: Data Preparation")
numerical_columns = ['Age', 'Years Employed', 'Income', 'Card Debt', 'Other Debt', 'DebtIncomeRatio']

# Handle missing values
if data['Defaulted'].isnull().sum() > 0:
    data['Defaulted'] = data['Defaulted'].fillna(data['Defaulted'].mode()[0])

# Normalize numerical columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

st.write("Data setelah normalisasi:")
st.write(data.head())

# Step 2: K-Means Clustering
st.write("## Step 2: K-Means Clustering")
clusters = st.sidebar.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)
kmeans = KMeans(n_clusters=clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[numerical_columns])

# Silhouette Score
silhouette_avg = silhouette_score(data[numerical_columns], data['Cluster'])
st.write(f"Silhouette Score untuk {clusters} cluster: {silhouette_avg:.2f}")

# Visualize clusters
st.write("### Visualisasi Cluster")
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=data, x='Income', y='DebtIncomeRatio', hue='Cluster', palette='viridis', ax=ax)
plt.title("K-Means Clustering")
plt.xlabel("Income")
plt.ylabel("Debt Income Ratio")
st.pyplot(fig)

st.write("Keterangan: Grafik di atas menunjukkan hasil clustering berdasarkan fitur 'Income' dan 'Debt Income Ratio'. Tiap warna merepresentasikan cluster yang berbeda.")

# Step 3: Logistic Regression
st.write("## Step 3: Logistic Regression")
X = data[numerical_columns]
y = data['Defaulted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.write("Keterangan: Confusion matrix di atas menunjukkan performa model Logistic Regression dalam memprediksi apakah pelanggan akan default atau tidak.")

# Evaluation Metrics
st.write("### Logistic Regression Evaluation")
st.write("Confusion Matrix:")
st.write(conf_matrix)

st.write("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Logistic Regression Coefficients
if st.sidebar.checkbox("Show Logistic Regression Coefficients"):
    coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": logreg.coef_[0]})
    st.write("### Logistic Regression Coefficients")
    st.write(coefficients)

    st.write("Keterangan: Tabel di atas menunjukkan kontribusi tiap fitur terhadap prediksi model Logistic Regression.")