from flask import Flask,request,jsonify
from flask_cors import CORS
import cosine

app = Flask(__name__)
CORS(app) 
        
@app.route('/', methods=['GET'])
def recommend_jobs():
    res = cosineword.recommend(request.args.get("java architect - denver, co - fulltime"))
    return jsonify(res)

if __name__=='__main__':
    app.run(port = 5000, debug = True)
