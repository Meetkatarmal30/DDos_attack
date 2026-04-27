from flask import Flask, render_template, jsonify
import sqlite3
import os

app = Flask(__name__)
DB_PATH = 'dashboard.db'

def get_db_connection():
    # If the database doesn't exist, it likely means realtime.py hasn't created it yet.
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Renders the main simple Dashboard page (using index.html)."""
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    """Serves the latest intrusion detection statistics required by dashboard scripts via AJAX."""
    conn = get_db_connection()
    if not conn:
        return jsonify({
            'total_processed': 0,
            'total_attacks': 0,
            'predictions': []
        })
        
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM stats WHERE id=1')
        stats_row = c.fetchone()
        
        c.execute('SELECT * FROM predictions ORDER BY id DESC LIMIT 10')
        predictions_rows = c.fetchall()
        
        conn.close()
        
        if stats_row:
            return jsonify({
                'total_processed': stats_row['total_processed'],
                'total_attacks': stats_row['total_attacks'],
                'predictions': [dict(row) for row in predictions_rows]
            })
    except Exception as e:
        print("Database read error:", e)
        
    return jsonify({
        'total_processed': 0,
        'total_attacks': 0,
        'predictions': []
    })

if __name__ == '__main__':
    # Running flask dashboard server
    print("starting Web Dashboard. Visit http://127.0.0.1:5000/")
    app.run(debug=True, port=5000, use_reloader=False)
