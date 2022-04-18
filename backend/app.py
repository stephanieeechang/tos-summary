from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={'/api/*': {'origins': '*'}}
)


@app.before_first_request
def load_models():
    pass


@app.route('/api/summarize', methods=['GET'])
def get_text_summary():
    pass


if __name__ == '__main__':
    app.run()
