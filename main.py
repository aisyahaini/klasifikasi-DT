import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

#membuat judul web
st.title(""" Aplikasi Classification Decision Tree Obesity \n """)

st.write("""Disini Saya Membuat Web Apps Klasifikasi Obesitas Menggunakan Algoritma Decision Tree 
         Untuk Meng-klasifikasi Apaakah Orang Tersebut Termasuk Kategori Obesitas Atau Tidak. 
         Kemudian Saya Menggunakan Dataset Obesitas Yang Saya Ambil dari Kaggle : https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset 
         Dari Dataset Tersebut Berjumlah 108 Data dan terdapat Atribut Age, Gender, Height, Weight, BMI, dan Label. Kemudian Saya Menggunakan Referensi Soruce Code dari : 
         https://www.kaggle.com/code/professordantez/obesity-classification """)

def load_csv():
    file_path = st.file_uploader("Upload CSV file", type=["csv"])
    if file_path is not None:
        df = pd.read_csv(file_path)
        return df
    return None

def plot_label_distribution(df):
    # Plotting the distribution of labels
    label_counts = df["Label"].value_counts()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(label_counts, labels=["Underweight", "Normal", "Overweight", "Obese"], autopct="%0.1f%%", startangle=90)
    ax.set_title("Distribusi Kategori Berat Badan dari Dataset")
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig)

def plot_gender_weight_distribution(df):
    # Plotting the gender-weight distribution
    fig, ax = plt.subplots()
    sns.countplot(x=df["Gender"], hue=df['Label'], ax=ax)
    
    # Adding labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title("Grafik Untuk Atribut Gender", size=15)
    
    st.pyplot(fig)

def plot_age_weight_distribution(df):
    # Plotting the age-weight distribution
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size of the plot
    sns.countplot(x=df["Age"], hue=df['Label'], ax=ax)
    
    # Adding labels to the bars with adjusted font size
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=6, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title("Grafik Untuk Atribut Age", size=15)
    
    st.pyplot(fig)
    
def plot_height_weight_distribution(df):
    # Plotting the gender-weight distribution
    fig, ax = plt.subplots()
    sns.countplot(x=df["Height"], hue=df['Label'], ax=ax)
    
    # Adding labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title("Grafik Untuk Atribut Height", size=15)
    
    st.pyplot(fig)
    
def plot_weight_weight_distribution(df):
    # Plotting the gender-weight distribution
    fig, ax = plt.subplots()
    sns.countplot(x=df["Weight"], hue=df['Label'], ax=ax)
    
    # Adding labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title("Grafik Untuk Atribut Weight", size=15)
    
    st.pyplot(fig)
    
def plot_bmi_weight_distribution(df):
    # Plotting the gender-weight distribution
    fig, ax = plt.subplots()
    sns.countplot(x=df["BMI"], hue=df['Label'], ax=ax)
    
    # Adding labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title("Grafik Untuk Atribut BMI", size=15)
    st.pyplot(fig)

def plot_heatmap_weight_classification():
    # Contoh DataFrame dengan nilai korelasi antar atribut
    data = {
        'Age': np.random.rand(10),
        'Gender': np.random.rand(10),
        'Height': np.random.rand(10),
        'Weight': np.random.rand(10),
        'BMI': np.random.rand(10),
    }
    df_corr = pd.DataFrame(data)

    # Menciptakan heatmap dengan transparansi
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')

    # Menambahkan label pada setiap sel
    for i, index in enumerate(df_corr.corr().columns):
        plt.text(i + 0.5, i + 0.5, f'{df_corr.corr()[index][index]:.2f}', ha='center', va='center', color='white')

    # Menampilkan plot
    #fig, ax = plt.subplots()
    plt.title("Heatmap", size=15)
    st.pyplot(plt)


#@st.cache(allow_output_mutation=True)
def preprocess_data(df):
    df["weight_classification"] = df["Label"].apply(lambda label: get_label(label))
    df["gender_classification"] = df["Gender"].apply(classify_gender)
    return df

def train_decision_tree_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    return clf, X_test, y_test

def evaluate_model(clf, X_test, y_test):
    # Evaluate the model
    weight_prediction = clf.predict(X_test)
    accuracy = accuracy_score(y_test, weight_prediction, normalize=True)
    
    # Display the accuracy in the Streamlit app
    st.subheader(f"### Confusion Matrix dan Akurasi Model:")
    #st.write(f"{accuracy:.2%}")
    
    y_pred = clf.predict(X_test)

    # Create a confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix using Seaborn heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    
    # Prediksi model
    y_pred = clf.predict(X_test)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Akurasi
    accuracy = accuracy_score(y_test, y_pred)
    # Presisi
    precision = precision_score(y_test, y_pred, average='weighted')
    # Recall
    recall = recall_score(y_test, y_pred, average='weighted')
    # F1-score
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Menampilkan hasil
    #st.write("Confusion Matrix:")
    #st.write(cm)
    #st.write(f"Akurasi Model : {accuracy:.2%}")
    #st.write(f"Presisi Model: {precision:.2%}")
    #st.write(f"Recall Model: {recall:.2%}")
    #st.write(f"F1-score Model: {f1:.2%}")
    
    # Konversi skor ke format persen
    accuracy_percent = f"{accuracy:.2%}"
    precision_percent = f"{precision:.2%}"
    recall_percent = f"{recall:.2%}"
    f1_percent = f"{f1:.2%}"

    # Membuat DataFrame untuk menampung metrik evaluasi
    metrics_data = {
        'Metric': ['Akurasi', 'Presisi', 'Recall', 'F1-score'],
        'Skor': [accuracy_percent, precision_percent, recall_percent, f1_percent]
    }

    metrics_df = pd.DataFrame(metrics_data)

    # Menampilkan tabel metrik evaluasi
    #st.table(metrics_df)
    st.write("### Akurasi Model :")
    st.write(f"{accuracy:.2%}")
    
#@st.cache(allow_output_mutation=True)
def single_predict(clf, class_mapping):
    st.title("Single Predict Classification Decision Tree Obesity")

    class_mapping = {
    1: "Underweight",
    2: "Normal Weight",
    3: "Overweight",
    4: "Obese"
}
    
    # Input fields for new attributes
    new_age = st.number_input("Masukkan Umur :")
    new_gender = st.selectbox("Pilih Jenis Kelamin:", ["Perempuan", "Laki-Laki"])
    new_height = st.number_input("Masukkan Tinggi Badan :")
    new_weight = st.number_input("Masukkan Berat Badan :")
    new_bmi = st.number_input("Masukkan BMI:")

    # Convert gender to numerical value
    new_gender = 1 if new_gender == "Perempuan" else 2

    # Create a list of new attributes
    new_attributes = [new_age, new_gender, new_height, new_weight, new_bmi]
    new_attributes = list(map(float, new_attributes))
    
    # Reshape the list to a NumPy array
    new_attributes_array = np.array(new_attributes).reshape(1, -1)

    # Create a DataFrame to display the array with labels
    attribute_labels = ["Umur", "Gender", "Tinggi Badan", "Berat Badan", "BMI"]
    new_attributes_df = pd.DataFrame(new_attributes_array, columns=attribute_labels)

    # Map the gender to the actual label
    new_attributes_df["Gender"] = new_attributes_df["Gender"].map({1: "Perempuan", 2: "Laki-Laki"})
    
    st.subheader("Array Untuk Atribut Prediksi Baru:")
    st.write(new_attributes_df)

    # Make a prediction using the model
    prediction = clf.predict(new_attributes_array)

    #st.write("Raw Prediction:")
    #st.write(prediction)

    # Map the predicted class to the actual label
    predicted_label = class_mapping.get(prediction[0], "Unknown")

    st.subheader("Hasil Klasifikasi Untuk Berat Badan :")
    st.write(predicted_label)

    # Display the result in the Streamlit app
    #st.write(f"Hasil Klasifikasi Untuk Berat Badan : {predicted_label}")


def get_label(text):
    classification = 0
    if text == "Underweight":
        classification = 1
    elif text == "Normal Weight":
        classification = 2
    elif text == "Overweight":
        classification = 3
    else:
        classification = 4
    return classification

def classify_gender(gender):
    classification = 0
    if gender == "Female":
        classification = 1
    else:
        classification = 2
    return classification

def main():
    global clf, class_mapping # Declare clf as a global variable

    st.sidebar.title("Aplikasi Klastering Decision Tree Obesity")

    menu = ["Input File CSV", "Single Predict"]
    choice = st.sidebar.selectbox("Pilih Menu", menu)

    if choice == 'Input File CSV':
        # Load CSV file
        df = load_csv()

        if df is not None:
            # Display DataFrame
            st.write("### Dataset :")
            st.write(df)

            # Plot label and gender-weight distribution
            st.subheader("Grafik Distribusi Kategori Label Berat Badan dari Dataset")
            plot_label_distribution(df)
            st.subheader("Grafik Distribusi Per Atribut dari Dataset")
            plot_age_weight_distribution(df)
            plot_gender_weight_distribution(df)
            plot_height_weight_distribution(df)
            plot_weight_weight_distribution(df)
            plot_bmi_weight_distribution(df)
            
            st.subheader("Heatmap Untuk Antar Atribut")
            plot_heatmap_weight_classification()

            # Preprocess data
            df = preprocess_data(df)
            features = ["Age", "gender_classification", "Height", "Weight", "BMI"]
            X = df[features]
            y = df["weight_classification"]
            
            # Train decision tree model
            clf, X_test, y_test = train_decision_tree_model(X, y)
            
            # Fit LabelEncoder and get class mapping
            le = LabelEncoder()
            le.fit(y)
            class_mapping = {i: label for i, label in enumerate(le.classes_)}
            
            # Evaluate model
            evaluate_model(clf, X_test, y_test)

            st.subheader("Pohon Keputusan")
             # Plot decision tree
            fig, ax = plt.subplots(figsize=(7, 9), dpi=100)
            plot_tree(clf, feature_names=X.columns, class_names=sorted(df["Label"].unique()), filled=True, ax=ax)
            st.pyplot(fig)
            
            # Menampilkan pohon keputusan dalam bentuk teks
            st.subheader("Pohon Keputusan (Format Teks):")
            tree_text = export_text(clf, feature_names=features, class_names=sorted(df["Label"].unique()))
            st.text(tree_text)
            
    elif choice == 'Single Predict':
        if 'clf' not in globals() or clf is None:
            # Load CSV file
            df = load_csv()

            if df is not None:
                # Display DataFrame
                #st.write("Dataset :")
                #st.write(df.head())

                # Plot label and gender-weight distribution
                #plot_label_distribution(df)
                #plot_gender_weight_distribution(df)

                # Preprocess data
                df = preprocess_data(df)
                features = ["Age", "gender_classification", "Height", "Weight", "BMI"]
                X = df[features]
                y = df["weight_classification"]

                # Train decision tree model
                clf, _, _ = train_decision_tree_model(X, y)
                
                # Fit LabelEncoder and get class mapping
            le = LabelEncoder()
            le.fit(y)
            class_mapping = {i: label for i, label in enumerate(le.classes_)}
        
        single_predict(clf, class_mapping)
        
if __name__ == "__main__":
    main()

