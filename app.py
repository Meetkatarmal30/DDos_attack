from flask import Flask, render_template, jsonify
import sqlite3
import os

app = Flask(__name__)
DB_PATH = 'dashboard.db'

def get_db():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    conn = get_db()
    if not conn:
        return jsonify({
            'total_processed': 0,
            'total_attacks': 0,
            'attack_rate': 0,
            'predictions': [],
            'status': 'waiting'
        })
    try:
        c = conn.cursor()
        c.execute('SELECT * FROM stats WHERE id=1')
        s = c.fetchone()
        c.execute(
            'SELECT * FROM predictions ORDER BY id DESC LIMIT 10'
        )
        preds = c.fetchall()
        conn.close()

        total = s['total_processed'] if s else 0
        attacks = s['total_attacks'] if s else 0
        rate = round((attacks / total * 100), 1) if total > 0 else 0

        return jsonify({
            'total_processed': total,
            'total_attacks': attacks,
            'attack_rate': rate,
            'predictions': [dict(r) for r in preds],
            'status': 'running' if total > 0 else 'waiting'
        })
    except Exception as e:
        print(f"DB error: {e}")
        if conn:
            conn.close()
        return jsonify({
            'total_processed': 0,
            'total_attacks': 0,
            'attack_rate': 0,
            'predictions': [],
            'status': 'error'
        })

if __name__ == '__main__':
    print("Dashboard starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000, use_reloader=False)