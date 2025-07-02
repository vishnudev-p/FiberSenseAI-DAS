import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from pathlib import Path
import os
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import base64

app = Flask(__name__)

# Step 1: Preprocess DAS Data
def preprocess_das_data(file_path, sample_rate=1000, threshold=0.85):
    df = pd.read_csv(file_path)
    time_col = 'Time' if 'Time' in df.columns else df.columns[0]
    if pd.api.types.is_numeric_dtype(df[time_col]) or df[time_col].isnull().any():
        df['Time'] = pd.date_range(start='2023-01-01 00:00:00', periods=len(df), freq=f'{1000 // sample_rate}ms')
    else:
        df['Time'] = pd.to_datetime(df[time_col])
    df['Time_seconds'] = (df['Time'] - df['Time'].min()).dt.total_seconds()
    channel_columns = [col for col in df.columns if col.startswith('Channel_')]
    if not channel_columns:
        channel_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in [time_col]]
    df[channel_columns] = df[channel_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df, channel_columns

# Step 2: Train Model
def train_model(labeled_data_path):
    df_labeled = pd.read_csv(labeled_data_path)
    df_labeled['Channel_number'] = df_labeled['Channel'].str.extract('(\d+)').astype(int)
    X = df_labeled[['Time_seconds', 'Channel_number']]
    y = df_labeled['Predicted_Label'].map({'shaker': 1, 'noise': 0})
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    scaler = StandardScaler()
    X_resampled_scaled = scaler.fit_transform(X_resampled)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_resampled_scaled, y_resampled)
    return knn, scaler

# Step 3: Detect Anomalies Efficiently
def detect_and_predict(df, channel_columns, threshold, knn, scaler):
    stacked_df = df[['Time', 'Time_seconds'] + channel_columns].melt(
        id_vars=['Time', 'Time_seconds'],
        var_name='Channel',
        value_name='Value'
    )
    anomaly_df = stacked_df[stacked_df['Value'] > threshold].copy()
    if not anomaly_df.empty:
        anomaly_df['Channel_number'] = anomaly_df['Channel'].str.replace('Channel_', '', regex=False).astype(int)
        X_new = anomaly_df[['Time_seconds', 'Channel_number']]
        X_scaled = scaler.transform(X_new)
        y_pred = knn.predict(X_scaled)
        anomaly_df['Predicted_Label'] = pd.Series(y_pred).map({1: 'shaker', 0: 'noise'})
    else:
        anomaly_df = pd.DataFrame(columns=['Time', 'Channel', 'Time_seconds', 'Predicted_Label'])
    return anomaly_df

#Step 4: Generate Output (modified for chart data)
def generate_output(anomaly_df, output_prefix):
    anomaly_counts = anomaly_df['Channel'].value_counts().sort_index()
    peak_channel = anomaly_counts.idxmax() if not anomaly_counts.empty else "N/A"
    peak_count = anomaly_counts.max() if not anomaly_counts.empty else 0

    # Prepare data for Chart.js
    labels = anomaly_counts.index.tolist()
    data = anomaly_counts.values.tolist()

    report = f"=== Anomaly Detection Report for {output_prefix} ===\n"
    report += f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S IST')}\n"
    report += f"Total Anomalies Detected: {len(anomaly_df)}\n"
    report += f"Spatial Analysis: Highest anomaly count at Channel {peak_channel} with {peak_count} anomalies.\n"
    report += "Classification Summary:\n"
    report += anomaly_df['Predicted_Label'].value_counts().to_string() + "\n"
    report += "Observations:\n- Anomalies detected above threshold may indicate shaker activity or noise.\n"
    report += f"- Spatial distribution suggests the shaker may be near Channel {peak_channel}, pending validation.\n"
    report += "Recommendations:\n- Adjust threshold if fewer/more anomalies are expected.\n- Refine labels or retrain with new data for better accuracy.\n"

    base_path = Path("C:/Users/HP/Desktop/New folder (2)/flaks/das_flask_app")
    base_path.mkdir(parents=True, exist_ok=True)

    anomaly_csv = base_path / f"{output_prefix}_anomalies.csv"
    anomaly_txt = base_path / f"{output_prefix}_report.txt"
    anomaly_df.to_csv(anomaly_csv, index=False)
    with open(anomaly_txt, 'w') as f:
        f.write(report)

    return report, {"labels": labels, "data": data}, anomaly_csv

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if file and file.filename.endswith('.csv'):
                sample_rate = int(request.form.get('sample_rate', 1000))
                threshold = float(request.form.get('threshold', 0.85))

                file_path = Path("C:/Users/HP/Desktop/New folder (2)/flaks/das_flask_app/uploads") / file.filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file.save(file_path)

                df, channel_columns = preprocess_das_data(file_path, sample_rate, threshold)

                labeled_path = 'C:/Users/HP/Desktop/New folder (2)/flaks/das_flask_app/das_labeled_anomalies.csv'
                if not os.path.exists(labeled_path):
                    return jsonify({'error': 'Labeled data file not found'}), 500
                model, scaler = train_model(labeled_path)

                anomaly_df = detect_and_predict(df, channel_columns, threshold, model, scaler)

                output_prefix = file_path.stem
                report, chart_data, anomaly_csv = generate_output(anomaly_df, output_prefix)

                return jsonify({
                    'report': report,
                    'chart_data': chart_data,
                    'csv_path': str(anomaly_csv),
                    'report_path': str(Path(file_path).with_suffix('.txt')),
                    'plot_path': str(Path(file_path).with_name(output_prefix + '_anomaly_distribution.png'))
                })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)