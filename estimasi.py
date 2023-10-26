import streamlit as st
import pickle

model = pickle.load(open('estimasi_harga_tiket.sav', 'rb'))

st.title('Estimasi Harga Tiket Pesawat Di India')
st.write('''
    Ini adalah aplikasi untuk memprediksi harga tiket pesawat di India.
    Silahkan masukkan data-data berikut ini:
''')

# Input
st.sidebar.header('Masukkan Data')

airline = st.sidebar.selectbox('Airline', ('AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara'))
#convert to numeric using switch case
if airline == 'AirAsia':
    airline = 0
elif airline == 'Air_India':
    airline = 1
elif airline == 'GO_FIRST':
    airline = 2
elif airline == 'Indigo':
    airline = 3
elif airline == 'SpiceJet':
    airline = 4
elif airline == 'Vistara':
    airline = 5

source = st.sidebar.selectbox('Source', ('Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'))  
#same here
if source == 'Bangalore':
    source = 0
elif source == 'Chennai':
    source = 1
elif source == 'Delhi':
    source = 2
elif source == 'Hyderabad':
    source = 3
elif source == 'Kolkata':
    source = 4
elif source == 'Mumbai':
    source = 5

departure_time = st.sidebar.selectbox('Departure Time', ('Afternoon', 'Early Morning', 'Evening', 'Late_night', 'Morning', 'Night'))

if departure_time == 'Afternoon':
    departure_time = 0
elif departure_time == 'Early Morning':
    departure_time = 1
elif departure_time == 'Evening':
    departure_time = 2
elif departure_time == 'Late_night':
    departure_time = 3
elif departure_time == 'Morning':
    departure_time = 4
elif departure_time == 'Night':
    departure_time = 5
  
stops = st.sidebar.selectbox('Stops', ('0', '1', '2'))

if stops == '0':
    stops = 0
elif stops == '1':
    stops = 1
elif stops == '2':
    stops = 2

arrival_time = st.sidebar.selectbox('Arrival Time', ('Afternoon', 'Early Morning', 'Evening', 'Late_night', 'Morning', 'Night'))

if arrival_time == 'Afternoon':
    arrival_time = 0
elif arrival_time == 'Early Morning':
    arrival_time = 1
elif arrival_time == 'Evening':
    arrival_time = 2
elif arrival_time == 'Late_night':
    arrival_time = 3
elif arrival_time == 'Morning':
    arrival_time = 4
elif arrival_time == 'Night':
    arrival_time = 5

destination = st.sidebar.selectbox('Destination', ('Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'))

if destination == 'Bangalore':
    destination = 0
elif destination == 'Chennai':
    destination = 1
elif destination == 'Delhi':
    destination = 2
elif destination == 'Hyderabad':
    destination = 3
elif destination == 'Kolkata':
    destination = 4
elif destination == 'Mumbai':
    destination = 5

class_ = st.sidebar.selectbox('Class', ('Business', 'Economy'))

if class_ == 'Business':
    class_ = 0
elif class_ == 'Economy':
    class_ = 1

duration = st.number_input('Duration (in hour)', min_value=0.83, max_value=50.00, value=10.00)

days_to_departure = st.number_input('Days to Departure', min_value=1, max_value=49, value=10)

flight_number = st.number_input('Flight Number', min_value=1, max_value=10000, value=950) 

if st.button('Estimasi'):
    prediction = model.predict([[airline, source, departure_time, stops, arrival_time, destination, class_, duration, days_to_departure, flight_number]])
    st.write('''
        Harga tiket pesawat yang diestimasi adalah sebesar: INR {:,.2f}
    '''.format(prediction[0]))