from flask import Flask,request,jsonify
from flask_cors import CORS
import cosine
import logging


app = Flask(__name__)
CORS(app) 
        
@app.route('/jobs', methods=['GET'])
def recommend_jobs():
    res = cosine.recommend(request.args.get('title'))
    return jsonify(res)

if __name__=='__main__':
    app.run(port = 5000, debug = True)
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)
