# Laporan Proyek Machine Learning
### Nama : Nurlisa Widyaningsih
### Nim : 211351108
### Kelas : Pagi B

## Domain Proyek

India merupakan negara yang besar dengan banyak kota-kota besar. Web App ini memberikan layanan estimasi patokan harga tiket pesawat di India. Sangat cocok bagi orang yang sedang berada di India dan ingin pergi ke kota-kotanya, terutama bagi pelaksana business profesional ataupun hanya ingin berkunjung.

## Business Understanding

Dengan memasukkan beberapa komponen yang anda inginkan sebagai spesifikasi penerbangan anda, Web App ini bisa melakukan estimasi harga tiketnya, jadi tidak diperlukan lagi untuk mencari tiket-tiket yang sesuai di situs-situs travel. Anda cukup menggunakan Web App ini, mendapatkan estimasi harganya, lalu mencari tiket yang sesuai dengan harga estimasi yang diberikan agar harga yang anda dapatkan tidak terlalu tinggi.

### Problem Statements
- Tidak semua orang memiliki waktu untuk mencari tiket dengan harga yang tepat dengan cepat.

### Goals
- Dengan Web App ini diharapkan bisa menentukan harga yang tepat bagi tiket pesawat dengan spesifikasi yang sesuai juga.

## Data Understanding
Datasets yang saya gunakan untuk menyelesaikan masalah ini adalah Flight Price Prediction yang saya dapatkan dari website Kaggle. Datasets ini memiliki 12 kolom (11 kolom pada dataset yang sudah dibersihkan) dengan 300,261 baris data, berikut adalah link menuju ke datasets terkait <br>
[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction/data).

### Variabel-variabel pada Flight Price Prediction Dataset adalah sebagai berikut:
- Airline : Nama perusahaan maskapai penerbangan disimpan dalam kolom maskapai. Ini adalah fitur kategoris dengan 6 maskapai yang berbeda.
- Flight : Penerbangan menyimpan informasi mengenai kode penerbangan pesawat. Ini adalah fitur kategoris.
- Source City : Kota dari mana penerbangan berangkat. Ini adalah fitur kategoris dengan 6 kota unik.
- Departure Time : ni adalah fitur kategoris turunan yang diperoleh dengan mengelompokkan periode waktu ke dalam kelompok-kelompok. Ini menyimpan informasi tentang waktu keberangkatan dan memiliki 6 label waktu yang unik.
- Stops : Fitur kategoris dengan 3 nilai berbeda yang menyimpan jumlah transit antara kota asal dan tujuan.
- Arrival Time : Ini adalah fitur kategoris turunan yang dibuat dengan mengelompokkan interval waktu ke dalam kelompok-kelompok. Ini memiliki enam label waktu yang berbeda dan menyimpan informasi tentang waktu kedatangan.
- Destination City : Kota tempat pesawat akan mendarat. Ini adalah fitur kategoris dengan 6 kota unik.
- Class : Fitur kategoris yang berisi informasi tentang kelas tempat duduk; memiliki dua nilai berbeda: Bisnis dan Ekonomi.
- Duration : Fitur kontinu yang menampilkan total waktu yang dibutuhkan untuk perjalanan antara kota-kota dalam jam.
- Days Left : Ini adalah karakteristik turunan yang dihitung dengan mengurangkan tanggal perjalanan dengan tanggal pemesanan.
- Price : Variabel target menyimpan informasi harga tiket.

## Data Preparation
Teknik data preparation yang saya gunakan ada EDA dan karena datasetsnya sudah cukup bersih maka saya hanya akan mengubah data object menjadi data numerik agar nantinya bisa diproses oleh Linear Regression, langkah awal adalah memasukkan token kaggle, membuat folder kaggle lalu mendownload datasets yang diinginkan.
``` bash
from google.colab import files
files.upload()
```

``` bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

``` bash
!kaggle datasets download -d shubhambathwal/flight-price-prediction
```
Langkah seterusnya adalah mengekspor datasets yang telah diunduh ke dalam sebuah folder,
``` bash
!unzip flight-price-prediction.zip -d flight_prices
!ls flight_prices
```
Selanjutnya mengimpor semua library yang akan digunakan bagi proses data preparation ini,
``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
Disini terdapat 3 file csv, namun yang akan kita gunakan adalah Clean_Dataset.csv karena merupakan datasets yang sudah tidak ada nilai null ataupun nilai yang aneh-aneh,
``` bash
df = pd.read_csv("flight_prices/Clean_Dataset.csv")
df.head()
```
Seperti yang bisa dilihat, setelah menampilkan 5 data pertama dari datasets, terdapat satu kolom yang tidak diinginkan, yaitu kolom Unnamed: 0, mari hilangkan terlebih dahulu,
``` bash
df=df.drop(["Unnamed: 0"],axis=1)
```
Untuk berjaga-jaga kita akan periksa apakah datasetsnya ada nilai null dan/atau data duplikasi,
``` bash
df.isnull().sum()
df.duplicated().sum()
```
Selanjutnya mari periksa tipe data dari masing-masing kolom,
``` bash
df.info()
```
Mari periksa jumlah data yang tersedia pada datasets ini dengan kode berikut,
``` bash
df.describe()
```
Bisa dilihat bahwa terdapat 300,153 baris data yang tersedia untuk diproses menjadi model, selanjutnya mari periksa jumlah data yang terdapat dalam kolom "days_left", ini akan menghitung jumlah data dalam pengelompokan agar mudah dibaca,
``` bash
df["days_left"].value_counts()
```
Kita akan lakukan untuk kolom "duration" juga,
``` bash
df["duration"].value_counts()
```
Selanjutnya kita akan membuat variabel baru untuk menampung semua kolom kategorial (bertipe object),
``` bash
categorical_cols = ['airline','source_city','departure_time', 'arrival_time', 'destination_city','class']
```
Lalu mari lihat semua kolom yang ada pada datasets ini sekarang,
``` bash
df.columns      
```
Selanjutnya melihat kolerasi antara kolom harga dengan airline,
``` bash
plt.figure(figsize=(16,6))
plt.title("flight prices")
sns.barplot(x="airline", y="price",color="Orange",data=df);
plt.show()
```
![download](https://github.com/NurlisaWidya/estimasi-harga-tiket-pesawat/assets/148893422/4c011de5-4f97-4991-ac94-c02c9cc2fee0) <br>
Bisa dilihat bahwa airline Air_India dan Vistara memiliki harga tiket yang mahal dibandingkan airline lainnya, Selanjutnya mari lihat korelasi antar kolom yang memiliki tipe data numeric,
``` bash
sns.heatmap(df.corr(), annot = True)
plt.show()
```
![download](https://github.com/NurlisaWidya/estimasi-harga-tiket-pesawat/assets/148893422/4a828cc3-a75e-4372-8178-35170cf94de7)
Lalu mari konversikan kolom kolom yang memiliki tipe data object dengan kode dibawah ini,
``` bash
df[['airline_code', 'flight_number']] = df['flight'].str.split('-', n=1, expand=True)
df.drop("flight", axis = 1, inplace = True)
df.drop("airline_code", axis = 1, inplace = True)

df['flight_number'] = df['flight_number'].astype('int')
df.flight_number.value_counts()
```
Kita bisa menggunakan LabelEncoder untuk mengkonversinya dengan mudah, seperti berikut,
``` bash
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder, OrdinalEncoder

df['stops'] = df['stops'].replace({'one': 1,
                                   'zero': 0,
                                   'two_or_more': 2})
df[categorical_cols] = df[categorical_cols].apply(le.fit_transform)
sns.heatmap(df.corr(), annot = True)
plt.show()
```
![download](https://github.com/NurlisaWidya/estimasi-harga-tiket-pesawat/assets/148893422/69a42d2d-ecc2-4c00-8827-f51f5ef91fc5) <br>
Bisa terlihat disini bahwa korelasi antar kolom cukup tinggi. Selanjutnya kita akan membuat modelnya.

## Modeling
Untuk modeling ini saya menggunakan Linear Regression sebagai algorithm trainnya, langkah pertama yang harus dilakukan adalah mengimpor semua library yang akan digunakan nanti,
``` bash
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```
Lalu saya akan membuat variabel untuk menampung kolom-kolom fitur dan kolom targetnya, 
``` bash
x = df.drop('price', axis = 1)
y = df['price']
```
Lalu membuat training dan test menggunakan train_test_split dengan test 20% dan train 80%,
``` bash
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
```
Selanjutnya membuat model Linear Regression dan memasukkan hasil train yang tadi dibuat,
``` bash
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
```
Pembuatan model sudah selesai, mari lihat score yang didapatkan,
``` bash
score = model.score(x_test, y_test)
print(f"this has {score} of score")
```
Model dengan score 90.65% sangatlah bagus, mari coba prediksi dengan data dummy kita,
``` bash
data = np.array([[4, 2, 2, 0, 5, 5, 1, 2.17, 1, 8709]])
prediction = model.predict(data)
print('Estimasi harga tiket pesawat dalam rupees : ', prediction)
```
Hasilnya adalah 5299 rupees atau sekitar 900 ribuan. <br>
Mari mengekspor hasil modelnya untuk digunakan pada media lain,
``` bash
import pickle

filename = "estimasi_harga_tiket.sav"
pickle.dump(model,open(filename,'wb'))
```
## Evaluation
Matrik Evaluasi yang saya gunakan adalah r2_score, karena merupakan salah satu teknik yang terbaik untuk mengevaluasi seberapa cocoknya model regressi untuk data. Ini menunjukkan proporsi variasi dalam variabel tergantung yang dapat diprediksi dari variabel independen, berikut adalah kode yang saya gunakan,
``` bash
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

kf = KFold(shuffle=True, random_state=42, n_splits=5)

s = StandardScaler()
lr = LinearRegression()

estimator = Pipeline([("scaler", s),
                      ("regression", lr)])

predictions = cross_val_predict(estimator, x, y, cv=kf)
r2_score(y, predictions)
```
Score yang didapatkan adalah 90.63%, dan itu cukup baik.
## Deployment
[Estimasi Harga Tiket Pesawat](https://estimasi-harga-tiket-pesawat-lisaaa.streamlit.app/) <br>
![image](https://github.com/NurlisaWidya/estimasi-harga-tiket-pesawat/assets/148893422/0b7656c3-f6b9-45d2-b8e4-59e97085b25c)

