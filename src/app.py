from flask import Flask
from flask.helpers import send_file
from flask.wrappers import Request
from flask import request
app = Flask(__name__) 

@app.route('/')
def home():
    return "FIGSI"

@app.route('/get_image/')
def get_image():
    
    tag = request.args.get('tag')
    filename = 'GeneratedImages\\000270.jpg-{}.png'.format(tag)
  
    return send_file(filename, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True,port=80)