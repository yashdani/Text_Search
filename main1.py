"""@author: Yash Dani Student Id: 1001707349"""

import flask
from flask import Flask, render_template, request, jsonify
application = Flask(__name__)
application.debug = True
@application.route('/')
def first():
    print("First Page")
    return render_template('index.html')

@application.route('/search/', methods=['GET', 'POST'])
def search():
    text_input = request.args.get('query','')
    print(text_input)
    import display
    res, highlight_query =display.get_results(text_input)
    print(res[0:5])
    return render_template('display.html', result=res, highlight_q = highlight_query)

if __name__ == '__main__':
#    flask run
    #application.run(host='0.0.0.0',port=int()use_reloader=False)
   application.run(use_reloader=False)