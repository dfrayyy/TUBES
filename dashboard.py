import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score

# Streamlit setup
st.title("K-Means Clustering & Logistic Regression")
st.write("Visualisasi hasil analisis menggunakan algoritma K-Means dan Logistic Regression.")

# File uploader di sidebar
st.sidebar.title("Upload File")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV untuk dianalisis", type="csv")

# Load dataset
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    data = load_data(uploaded_file)

    # Sidebar for data exploration
    if st.sidebar.checkbox("Show Raw Data"):
        st.write("### Raw Dataset", data)

    # Step 1: Data Preparation
    st.write("## Step 1: Data Preparation")
    
    # Membuat kolom untuk memilih fitur-fitur yang akan dianalisis
    available_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_columns = st.sidebar.multiselect("Pilih Kolom untuk Analisis", available_columns, default=available_columns)

    # Pastikan pengguna memilih kolom
    if len(selected_columns) < 1:
        st.error("Pilih minimal satu kolom untuk analisis.")
    else:
        # Handle missing values
        if data['Defaulted'].isnull().sum() > 0:
            data['Defaulted'] = data['Defaulted'].fillna(data['Defaulted'].mode()[0])

        # Normalize selected numerical columns
        scaler = StandardScaler()
        data[selected_columns] = scaler.fit_transform(data[selected_columns])

        st.write("Data setelah normalisasi:")
        st.write(data.head())

        # Step 2: K-Means Clustering
        st.write("## Step 2: K-Means Clustering")
        
        # Elbow Method untuk memilih k optimal
        st.write("### Menentukan Jumlah Cluster Optimal dengan Elbow Method")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
            kmeans.fit(data[selected_columns])
            wcss.append(kmeans.inertia_)

        # Plot Elbow Method
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, 11), wcss, marker='o', color='b')
        ax.set_title('Elbow Method For Optimal k')
        ax.set_xlabel('Jumlah Cluster')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
        
        # Memilih k berdasarkan metode Elbow
        k_optimal = st.sidebar.slider("Pilih Jumlah Cluster (k) Secara Manual", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=k_optimal, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data[selected_columns])

        # Silhouette Score
        silhouette_avg = silhouette_score(data[selected_columns], data['Cluster'])
        st.write(f"Silhouette Score untuk {k_optimal} cluster: {silhouette_avg:.2f}")

        # Visualize clusters
        st.write("### Visualisasi Cluster")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=data, x=selected_columns[0], y=selected_columns[1], hue='Cluster', palette='viridis', ax=ax)
        plt.title("K-Means Clustering")
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        st.pyplot(fig)

        st.write("Keterangan: Grafik di atas menunjukkan hasil clustering berdasarkan fitur yang dipilih.")

        # Step 3: Logistic Regression
        st.write("## Step 3: Logistic Regression")
        X = data[selected_columns]
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

        # Visualisasi Logistic Regression Decision Boundary
        st.write("### Logistic Regression Decision Boundary")

        # Pastikan kolom-kolom yang dipilih untuk visualisasi ada di X
        if len(X.columns) >= 2:
            # Pilih dua fitur untuk divisualisasikan
            feature_1 = st.sidebar.selectbox("Pilih Fitur 1", X.columns, index=0)
            feature_2 = st.sidebar.selectbox("Pilih Fitur 2", X.columns, index=1)
        else:
            st.error("Data tidak memiliki cukup fitur untuk melakukan visualisasi.")

        if feature_1 and feature_2 and feature_1 != feature_2:
            fig, ax = plt.subplots(figsize=(6, 4))

            # Plotkan data
            sns.scatterplot(
                x=X[feature_1],
                y=X[feature_2],
                hue=y,
                palette='coolwarm',
                alpha=0.6,
                ax=ax
            )

            # Membuat grid untuk garis keputusan
            x_min, x_max = X[feature_1].min() - 1, X[feature_1].max() + 1
            y_min, y_max = X[feature_2].min() - 1, X[feature_2].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.01),
                np.arange(y_min, y_max, 0.01)
            )

            # Menyiapkan grid untuk prediksi (dua fitur yang dipilih)
            grid_points = np.c_[xx.ravel(), yy.ravel()]

            # Membuat salinan data untuk grid dengan semua fitur yang digunakan
            grid_data = pd.DataFrame(grid_points, columns=[feature_1, feature_2])

            # Menambahkan kolom lainnya dari data yang hilang di grid
            for column in [col for col in X.columns if col not in [feature_1, feature_2]]:
                grid_data[column] = X[column].mean()  # Mengisi kolom lain dengan nilai rata-rata

            # Prediksi probabilitas menggunakan model Logistic Regression
            Z = logreg.predict(grid_data)

            # Reshape prediksi agar sesuai dengan grid
            Z = Z.reshape(xx.shape)

            # Plotkan garis keputusan
            ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')

            plt.xlabel(feature_1)
            plt.ylabel(feature_2)
            plt.title("Decision Boundary Logistic Regression")
            st.pyplot(fig)

        else:
            st.write("Pilih dua fitur yang berbeda untuk visualisasi.")

        # Logistic Regression Coefficients
        if st.sidebar.checkbox("Show Logistic Regression Coefficients"):
            coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": logreg.coef_[0]})
            st.write("### Logistic Regression Coefficients")
            st.write(coefficients)

            st.write("Keterangan: Tabel di atas menunjukkan kontribusi tiap fitur terhadap prediksi model Logistic Regression.")

        # Visualisasi Heatmap Korelasi
        st.write("### Heatmap Korelasi Antar Fitur")
        corr_matrix = data[selected_columns].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax, linewidths=0.5)
        plt.title("Heatmap Korelasi Antar Fitur")
        st.pyplot(fig)
else:
    st.sidebar.write("Silakan unggah file CSV untuk memulai analisis.")
