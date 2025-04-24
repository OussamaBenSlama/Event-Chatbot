from flask import Flask, jsonify, request, render_template
from flask_cors import CORS  
from model import get_answer


app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')


# Enable CORS for all routes
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def talk():
    #retrieve query from request
    data = request.get_json()
    query = data.get('query')
    answer = get_answer(query)

    return jsonify({"data": answer}) 


if __name__ == '__main__':
    app.run(debug=True)
