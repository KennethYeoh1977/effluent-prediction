import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
from keras.regularizers import l1, l2
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf

# Disable GPU usage for TensorFlow
tf.config.set_visible_devices([], 'GPU')

def apply_moving_average(data_series, window_size=5):
    return data_series.rolling(window_size).mean()

def run_models(data):
    try:
        data['Normalized pH F/D'] = data['pH F/D']
        data['Normalized COD F/D'] = data['COD F/D']
        data['Normalized SS F/D'] = data['SS F/D']
        data['Normalized Zn F/D'] = data['Zn F/D']

        features = ['pH F/D', 'COD F/D', 'SS F/D']
        target = 'COD F/D'

        for column in features + [target]:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in the data.")

        data['COD F/D_MA'] = apply_moving_average(data[target])
        data = data.dropna()

        if data.shape[0] < 10:
            raise ValueError("Not enough samples in the data after preprocessing. Ensure the data has enough valid rows.")

        scaler = StandardScaler()
        features_with_ma = features + ['COD F/D_MA']
        normalized_data = scaler.fit_transform(data[features_with_ma])
        a = 20  
        b = 101  
        data[features_with_ma] = (b - a) * (normalized_data - normalized_data.min(axis=0)) / (normalized_data.max(axis=0) - normalized_data.min(axis=0)) + a

        X_train, X_test, y_train, y_test = train_test_split(data[features_with_ma], data[target], test_size=0.2, random_state=42)

        if X_train.shape[0] == 0:
            raise ValueError("Training set is empty after splitting. Adjust the test_size or ensure there are enough valid samples in the data.")

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l1(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.8)))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[MeanAbsoluteError()])

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(32)

        history = model.fit(train_dataset, epochs=20, steps_per_epoch=len(X_train) // 32, validation_data=(X_val, y_val))

        tdnn_mse_test, tdnn_mae_test = model.evaluate(X_test, y_test, verbose=0)
        tdnn_r2_test = r2_score(y_test, model.predict(X_test))

        st.write(f"TDNN Mean Squared Error (Testing): {tdnn_mse_test}")
        st.write(f"TDNN Mean Absolute Error (Testing): {tdnn_mae_test}")
        st.write(f"TDNN R-squared (Testing): {tdnn_r2_test}")

        data['Predicted COD F/D (TDNN)'] = model.predict(data[features_with_ma])
        overall_mae = mean_absolute_error(data[target], data['Predicted COD F/D (TDNN)'])

        last_date = data['Date'].iloc[-1]
        next_prediction_dates = [last_date + pd.Timedelta(days=1), last_date + pd.Timedelta(days=2)]
        last_features = data[features_with_ma].iloc[-1].values
        next_2_predictions = []
        for _ in range(2):
            next_prediction = model.predict(np.expand_dims(last_features, axis=0))[0][0]
            next_2_predictions.append(next_prediction)
            last_features = np.roll(last_features, -1, axis=0)
            last_features[-1] = next_prediction

        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['COD F/D'], label='Actual COD F/D')
        plt.plot(data['Date'], data['Predicted COD F/D (TDNN)'], label='Predicted COD F/D (TDNN)')
        plt.plot(data['Date'], data['COD F/D'].rolling(window=5).mean(), label='Moving Average (window=5)')
        plt.plot(next_prediction_dates, next_2_predictions, 'ro-', label='Next 2 Predictions')
        for i, (date, value) in enumerate(zip(next_prediction_dates, next_2_predictions)):
            plt.text(date, value, f"{value:.2f}", ha='center', va='bottom', fontsize=8)
        plt.text(data['Date'].iloc[int(len(data) * 0.8)], max(data['COD F/D']), f"MAE: {overall_mae:.2f}", fontsize=12, color='red')
        plt.xlabel('Date')
        plt.ylabel('COD F/D')
        plt.title('Actual vs Predicted COD F/D (TDNN) with Moving Average and Next 2 Predictions')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

        return data
    except Exception as e:
        st.error(f"Error processing the data: {e}")
        return None

st.title("Effluent Prediction")

st.markdown("""
Easiest way for reports. This application allows you to upload an Excel file, predict Industrial Effluent levels using Time-Delay Neural Networks (TDNN), and A.I generates reports. Designed by Ken
""")

uploaded_file = st.file_uploader("Upload your 'DATAKEN' format Excel file (max 50KB)", type=['xlsx'], key='1', accept_multiple_files=False)

if uploaded_file:
    if uploaded_file.size > 50000:
        st.error("File size exceeds 50KB limit. Please upload a smaller file.")
    else:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("Uploaded Data")
            st.dataframe(data)

            result = run_models(data)
            if result is not None:
                st.write("Prediction Results")
                st.dataframe(result)

                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")
