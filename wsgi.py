import os

from flask import jsonify

from app import create_app
from app.helper import aggregator

app = create_app()


@app.route('/get_currencies')
async def get_currencies():
    data = await aggregator()
    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)