import streamlit as st
import pandas as pd
import numpy as np 

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
import pickle

st.title("Gender Classification Dataset")
st.write("##### Nama  : Narulita Arien Pramesti ")
st.write("##### Nim   : 200411100065 ")
st.write("##### Kelas : Penambangan Data A ")

# inisialisasi data 
data = pd.read_csv("gender.csv")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description Data", "Preprocessing Data", "Modeling", "Implementation", "Profil"])

with tab1:

    st.subheader("Deskripsi Dataset")
    st.write("Dataset yang digunakan adalah gender classification dataset. Dataset ini memiliki 7 fitur yakni LongHair, ForeHeadWidth, ForeHeadHeight, NoseWide, NoseLong, LipsThin, NosetoLiplong")
    st.write("""
    Disini di jelaskan data-data yang ada dalam dataset tersebut seperti penjelasan dari setiap fitur yang
    ada dalam dataset tersebut :
    1. Long Hair : Kolom ini berisi 0 dan 1 di mana 1 adalah "rambut panjang" dan 0 adalah "rambut tidak panjang".
    2. ForeHeadWidth : Kolom ini dalam CM. Ini adalah lebar dahi.
    3. ForeHeadHeight : Ini adalah tinggi dahi dan dalam satuan Cm.
    4. NoseWide : Kolom ini berisi 0 dan 1 dimana 1 adalah "hidung lebar" dan 0 adalah "hidung tidak lebar".
    5. NoseLong : Kolom ini berisi 0 dan 1 di mana 1 adalah "hidung panjang" dan 0 adalah "hidung tidak panjang".
    6. LipsThin : Kolom ini berisi 0 dan 1 di mana 1 mewakili "bibir tipis" sedangkan 0 adalah "Bibir tidak tipis".
    7. NosetoLiplong : Kolom ini berisi 0 dan 1 dimana 1 mewakili "jarak jauh antara hidung dan bibir" sedangkan 0 adalah "jarak pendek antara hidung dan bibir".
    """)

    st.write("""
    ### Want to learn more?
    - Dataset [kaggel.com](https://www.kaggle.com/datasets/kannanaikkal/ecoli-uci-dataset)
    - Github Account [github.com](https://github.com/AliGhufron-28/datamaining)
    """)

    st.write(data)
    col = data.shape
    st.write("Jumlah Baris dan Kolom : ", col)
   
with tab2:
    st.subheader("Data Preprocessing")
    st.subheader("Data Asli")
    data = pd.read_csv("gender.csv")
    st.write(data)

    proc = st.checkbox("Normalisasi")
    if proc:

        # Min_Max Normalisasi
        from sklearn.preprocessing import MinMaxScaler
        df_for_minmax_scaler=pd.DataFrame(data, columns = ["long_hair","forehead_width_cm","forehead_height_cm","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"])
        df_for_minmax_scaler.to_numpy()
        scaler = MinMaxScaler()
        df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

        st.subheader("Hasil Normalisasi Min_Max")
        df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ["long_hair","forehead_width_cm","forehead_height_cm","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"])
        st.write(df_hasil_minmax_scaler)
        
        st.subheader("Tampil Data gender")
        df_gender = pd.DataFrame(data, columns = ['gender'])
        st.write(df_gender.head())

        st.subheader("Gabung Data")
        df_new = pd.concat([df_hasil_minmax_scaler,df_gender], axis=1)
        st.write(df_new)

        st.subheader("Drop fitur Gender")
        df_drop_site = df_new.drop(['gender'], axis=1)
        st.write(df_drop_site)

        st.subheader("Hasil Preprocessing")
        df_new = pd.concat([df_hasil_minmax_scaler,df_gender], axis=1)
        st.write(df_new)

with tab3:

    X=data.iloc[:,0:7].values
    y=data.iloc[:,7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    st.subheader("Pilih Model")
    model1 = st.checkbox("KNN")
    model2 = st.checkbox("Naive Bayes")
    model3 = st.checkbox("Random Forest")
    # model4 = st.checkbox("Ensamble Stacking")

    if model1:
        model = KNeighborsClassifier(n_neighbors=3)
        filename = "KNN.pkl"
        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma KNN : ",score)
    if model2:
        model = GaussianNB()
        filename = "GaussianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Naive Bayes GaussianNB : ",score)
    if model3:
        model = RandomForestClassifier(n_estimators = 100)
        filename = "RandomForest.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        st.write("Hasil Akurasi Algoritma Random Forest : ",score)
    #if model4:
     #   estimators = [
      #      ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
       #  ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        #]
       # model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
       # filename = "stacking.pkl"

       # model.fit(X_train,y_train)
      #  Y_pred = model.predict(X_test)

     #   score=metrics.accuracy_score(y_test,Y_pred)
      #  loaded_model = pickle.load(open(filename, 'rb'))
       # st.write("Hasil Akurasi Algoritma Ensamble Stacking : ",score)

with tab4:
    # Min_Max Normalisasi
    from sklearn.preprocessing import MinMaxScaler
    df_for_minmax_scaler=pd.DataFrame(data, columns = ["long_hair","forehead_width_cm","forehead_height_cm","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"])
    df_for_minmax_scaler.to_numpy()
    scaler = MinMaxScaler()
    df_hasil_minmax_scaler=scaler.fit_transform(df_for_minmax_scaler)

    df_hasil_minmax_scaler = pd.DataFrame(df_hasil_minmax_scaler,columns = ["long_hair","forehead_width_cm","forehead_height_cm","nose_wide","nose_long","lips_thin","distance_nose_to_lip_long"])

    df_gender = pd.DataFrame(data, columns = ['gender'])

    df_new = pd.concat([df_hasil_minmax_scaler,df_gender], axis=1)

    df_drop_site = df_new.drop(['gender'], axis=1)

    df_new = pd.concat([df_hasil_minmax_scaler,df_gender], axis=1)
    
    st.subheader("Parameter Inputan")
    # SEQUENCE_NAME = st.selectbox("Masukkan SEQUENCE_NAME : ", ("AAT_ECOLI","ACEA_ECOLI","ACEK_ECOLI","ACKA_ECOLI",
    # "ADI_ECOLI","ALKH_ECOLI","AMPD_ECOLI","AMY2_ECOLI","APT_ECOLI","ARAC_ECOLI"))
    LongHair = st.number_input("Masukkan Panjang Rambut :")
    ForeHeadWidth= st.number_input("Masukkan Lebar Dahi :")
    ForeHeadHeight = st.number_input("Masukkan Tinggi Dahi :")
    NoseWide = st.number_input("Masukkan Lebar Hidung :")
    NoseLong = st.number_input("Masukkan Panjang Hidung:")
    LipsThin = st.number_input("Masukkan ketebalan bibir :")
    NosetoLiplong = st.number_input("Masukkan Jarak hidung dan bibir :")
    # inisialisasi model algoritma yang digunakan
    # algoritma = st.selectbox(
    #     "Pilih Model",
    #     ("KNN", "Naive Bayes", "Random Forest","Stacking")
    # )
    hasil = st.button("Cek Klasifikasi")

    # Memakai yang sudah di preprocessing
    X=df_new.iloc[:,1:7].values
    y=df_new.iloc[:,7].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=0)

    if hasil:
        # if algoritma == "KNN":
        #     model = KNeighborsClassifier(n_neighbors=3)
        #     filename = "KNN.pkl"
        # elif algoritma == "Naive Bayes":
        #     model = GaussianNB()
        #     filename = "gaussianNB.pkl"
        # elif algoritma == "Random Forest":
        #     model = RandomForestClassifier(n_estimators = 100)
        #     filename = "RandomForest.pkl"
        # else:
        #     estimators = [
        #         ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
        #         ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        #         ]
        #     model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        #     filename = "stacking.pkl"
        model = GaussianNB()
        filename = "GaussianNB.pkl"

        model.fit(X_train,y_train)
        Y_pred = model.predict(X_test)

        score=metrics.accuracy_score(y_test,Y_pred)
        loaded_model = pickle.load(open(filename, 'rb'))
        
        dataArray = [LongHair, ForeHeadWidth, ForeHeadHeight, NoseWide, NoseLong, LipsThin, NosetoLiplong ]
        pred = loaded_model.predict([dataArray])

        st.success(f"Prediksi Hasil Klasifikasi : {pred[0]}")
        #st.write(f"Algoritma yang digunakan adalah = Random Forest Algorithm")
        #st.success(f"Hasil Akurasi : {score}")

with tab5:
    st.subheader("Profil Mahasiswa")
    st.write("""
    \nNama   : Narulita Arien Pramesti 
    \nNIM    : 200411100065
    \nKelas  : Penambangan Data A
    \nDisini saya mengerjakan tugas akhir dari mata kuliah Penambangan Data.
    \nUntuk tugas akhir saya mencoba melakukan analisis pada dataset yaitu Gender Classification atau Klasifikasi Jenis Kelamin.
    \n**Email**     : narulita.arien@gmail.com
    \n**Github**    : narulitaap
    \n**Instagram** : narulitaa.ap
    \n Terimakasih , BYEEE
    """)
