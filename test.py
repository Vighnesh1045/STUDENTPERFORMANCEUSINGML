from flask import Flask
test= Flask(__name__)

@test.route("/")
def hello():
    return "<p>hello</p>"