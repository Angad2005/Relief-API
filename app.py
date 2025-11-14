#!/usr/bin/env python3

"""
Full-stack AI Sensor Validation Backend
- SQLite DB recreated on every startup (for Render compatibility)
- Background threads: data injector + anomaly validator
- REST API for Vercel-hosted React dashboard
- CORS enabled for cross-origin requests
"""

import os
import threading
import time
import random
import numpy as np
import sqlite3
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.ensemble import IsolationForest

# --- Configuration ---
DB_FILE = "sensor_data.db"
TABLE_NAME = "mq2_data"
RESET_INTERVAL_SECONDS = 900  # Reset DB every 15 minutes

# --- Flask App Setup ---
app = Flask(__name__)
# Enable CORS for Vercel frontend (restrict in production if needed)
CORS(app, origins=[
    "https://your-frontend.vercel.app",   # Replace with your actual Vercel domain
    "http://localhost:3000",              # For local development
])

# --- Database Initialization ---
def setup_database():
    """Initialize a fresh SQLite database with table and seed data."""
    print("🔄 Initializing database...")
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("🗑️  Removed existing database file.")

    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    cursor = conn.cursor()

    # Create table
    cursor.execute(f"""
        CREATE TABLE {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sensor_value REAL,
            is_valid INTEGER DEFAULT NULL
        );
    """)
    print(f"✅ Table `{TABLE_NAME}` created.")

    # Generate initial dataset: 1000 normal + 30 anomalies
    normal_data = np.random.normal(loc=150, scale=25, size=1000)
    normal_data = np.clip(normal_data, 0, None)  # No negative values
    anomalies = np.concatenate([
        np.full(15, 900.0),  # High spikes
        np.full(15, 0.0)     # Flatlines
    ])
    all_values = np.concatenate([normal_data, anomalies]).tolist()
    random.shuffle(all_values)

    # Insert data
    cursor.executemany(f"INSERT INTO {TABLE_NAME} (sensor_value) VALUES (?)", [(v,) for v in all_values])
    conn.commit()
    conn.close()
    print(f"✅ Seeded database with {len(all_values)} records.")

# --- Background Thread: Data Injector ---
def data_injector():
    """Continuously inject simulated sensor readings every 1 second."""
    print("💉 Data injector thread started.")
    start_time = time.time()
    while True:
        try:
            current_time = time.time()
            elapsed = current_time - start_time

            # Reset database every 15 minutes
            if elapsed >= RESET_INTERVAL_SECONDS:
                print("⏰ 15-minute interval reached. Reinitializing database...")
                setup_database()
                start_time = time.time()
                continue

            # Generate reading: 95% normal, 5% anomaly
            if random.random() < 0.05:
                value = random.choice([0.0, 900.0])
                print(f"🚨 Injecting ANOMALY: {value}")
            else:
                value = max(0.0, np.random.normal(loc=150, scale=25))
                print(f"📈 Injecting normal reading: {value:.2f}")

            # Insert into DB
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            conn.execute(f"INSERT INTO {TABLE_NAME} (sensor_value) VALUES (?)", (value,))
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"⚠️  Injector error: {e}")

        time.sleep(1)

# --- Background Thread: Anomaly Validator ---
def validator_loop():
    """Periodically validate unprocessed records using Isolation Forest."""
    print("🔍 Validator thread started.")
    while True:
        try:
            conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            df = pd.read_sql_query(
                f"SELECT id, sensor_value FROM {TABLE_NAME} WHERE is_valid IS NULL",
                conn
            )
            if not df.empty:
                # Run Isolation Forest
                model = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)
                predictions = model.fit_predict(df[['sensor_value']])
                df['is_valid'] = np.where(predictions == 1, 1, 0)

                # Update DB
                updates = [(int(row['is_valid']), int(row['id'])) for _, row in df.iterrows()]
                conn.executemany(f"UPDATE {TABLE_NAME} SET is_valid = ? WHERE id = ?", updates)
                conn.commit()
                anomaly_count = (df['is_valid'] == 0).sum()
                print(f"✅ Validated {len(df)} records. Flagged {anomaly_count} anomalies.")
            conn.close()
        except Exception as e:
            print(f"⚠️  Validator error: {e}")

        time.sleep(60)  # Run every 60 seconds

# --- API Endpoint for Dashboard ---
@app.route('/api/dashboard-data')
def dashboard_data():
    """Return summary stats and latest anomalies for the frontend."""
    try:
        conn = sqlite3.connect(DB_FILE, check_same_thread=False)

        # Fetch summary stats
        stats_query = f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN is_valid = 1 THEN 1 ELSE 0 END) AS valid,
                SUM(CASE WHEN is_valid = 0 THEN 1 ELSE 0 END) AS invalid,
                SUM(CASE WHEN is_valid IS NULL THEN 1 ELSE 0 END) AS unprocessed
            FROM {TABLE_NAME};
        """
        stats_df = pd.read_sql_query(stats_query, conn)
        stats = stats_df.iloc[0].to_dict()
        # Convert NaN to 0 and ensure integers
        stats = {
            k: int(v) if pd.notna(v) and v is not None else 0
            for k, v in stats.items()
        }

        # Fetch latest 100 anomalies
        anomalies_query = f"""
            SELECT id, timestamp, sensor_value
            FROM {TABLE_NAME}
            WHERE is_valid = 0
            ORDER BY timestamp DESC
            LIMIT 100;
        """
        anomalies_df = pd.read_sql_query(anomalies_query, conn)
        anomalies = anomalies_df.to_dict(orient='records')
        conn.close()

        return jsonify({
            "stats": stats,
            "anomalies": anomalies
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Startup Hook (for production) ---
def start_background_threads():
    """Start injector and validator threads once."""
    # Check if threads are already running to avoid duplicates
    if not any("injector" in t.name for t in threading.enumerate()):
        print("Starting data injector thread...")
        threading.Thread(target=data_injector, daemon=True, name="injector").start()
    if not any("validator" in t.name for t in threading.enumerate()):
        print("Starting validator thread...")
        threading.Thread(target=validator_loop, daemon=True, name="validator").start()

# --- Application Initialization ---
# This code runs ONCE when the application module is imported
# (both for local dev and for production servers like Gunicorn on Render).
# This replaces the deprecated `@app.first_request_hook`.
print("🚀 Initializing database and starting background threads...")
setup_database()
start_background_threads()
print("✅ Application initialized.")

# --- Main entry (for local dev) ---
if __name__ == "__main__":
    # The init functions are already called above.
    # This block is now only for running the local Flask dev server.
    print(f"--- Running local development server ---")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)