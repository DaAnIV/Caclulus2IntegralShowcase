from flask import Flask, render_template, request
app = Flask(__name__)


@app.route("/")
def main():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def Calculate():
 
    # read the posted values from the UI
    _function = request.form['function']
    print(_function)

if __name__ == "__main__":
    app.run(debug=True)