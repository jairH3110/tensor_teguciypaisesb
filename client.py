import streamlit as st
import requests

SERVER_URL = 'https://custom-map-service-example.com/v1/models/custom-map-model:predict'

def get_predictions(inputs):
    predict_request = {'instances': inputs}
    response = requests.post(SERVER_URL, json=predict_request)
    
    if response.status_code == 200:
        prediction = response.json()
        return prediction
    else:
        st.error("Error al obtener predicciones. Por favor, verifica tus entradas e intenta de nuevo.")
        return None

def main():
    st.title('Predictor de Ubicaciones Geográficas')

    st.header('Coordenadas para uruguay')
    uruguay_lat = st.number_input('Ingrese la latitud de uruguay:', value=-34.6037)
    uruguay_lon = st.number_input('Ingrese la longitud de uruguay:', value=-58.3816)

    st.header('Coordenadas para barcelona')
    barcelona_lat = st.number_input('Ingrese la latitud de barcelona:', value=33.9391)
    barcelona_lon = st.number_input('Ingrese la longitud de barcelona:', value=67.7100)

    if st.button('Predecir'):
        inputs = [
            [uruguay_lon, uruguay_lat],
            [barcelona_lon, barcelona_lat]
        ]
        predictions = get_predictions(inputs)

        if predictions:
            st.write("\nPredicciones para uruguay:")
            st.write(predictions['predictions'][0])

            st.write("\nPredicciones para barcelona:")
            st.write(predictions['predictions'][1])

if _name_ == '_main_':
    main()