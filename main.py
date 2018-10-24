import flask
from flask import request, jsonify, make_response, abort

import imp
from sklearn.externals import joblib

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.testing = True




@app.route('/', methods=['GET'])
def home():
    return make_response(jsonify({'hello': 'hello world'}), 200)

@app.errorhandler(404)
def page_not_found(e):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/predictData', methods=['POST'])
def create_task():
    # if not request.json or not 'title' in request.json:
    #     abort(400)
    
    knn = joblib.load('filename.joblib')
    
    # resultArr = knn.predict([[request.json['time'], request.json['clippededges'], request.json['hipcenter'], request.json['spine'], request.json['shouldercenter'], request.json['head'], request.json['shoulderleft'], request.json['elbowleft'], request.json['wristleft'], request.json['handleft'], request.json['shoulderright'], request.json['elbowright'], request.json['wristright'], request.json['handright'], request.json['hipleft'], request.json['kneeleft'], request.json['ankleleft'], request.json['footleft'], request.json['hipright'], request.json['kneeright'],request.json['ankleright']]])
    resultArr = knn.predict([[0.0200153, -0.3014402, -0.3008801, -0.2884384,-0.2709138,-0.4670434,-0.5239125,-0.5501519,-0.5606456,-0.1176529,-0.07766194,-0.05077618,-0.04127074,-0.3825685,-0.4515708,-0.4349683,-0.4293025,-0.2254354,-0.2188429,-0.2053359,-0.227809]])
    result = str(resultArr[0])
    return jsonify({'type': result}), 200


if __name__ == '__main__':
  app.run()