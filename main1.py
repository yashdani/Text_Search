"""@author: Yash Dani Student Id: 1001707349"""

import flask
from flask import Flask, render_template, request, jsonify
application = Flask(__name__)
application.debug = True
@application.route('/')
def first():
    print("Index Page")
    return render_template('index.html')

@application.route('/search/', methods=['GET', 'POST'])
def search():
    text_input = request.args.get('query','')
    print(text_input)
    import search_query
    res, highlight_query = search_query.get_results(text_input)
    print(res[0:10])
    return render_template('display.html', result=res, highlight_q=highlight_query)

@application.route('/classify/', methods=['GET', 'POST'])
def classify():
    text_input = request.args.get('query','')
    import classify_query
    result = classify_query.get_results(text_input)
    # data = {'results': query_classifier.get_results(classify_query)}
    # data = jsonify(data)
    print(result[0:10])
    return render_template('display_classify.html', result=result)

if __name__ == '__main__':
#    flask run
    #application.run(host='0.0.0.0',port=int()use_reloader=False)
   application.run(use_reloader=False)