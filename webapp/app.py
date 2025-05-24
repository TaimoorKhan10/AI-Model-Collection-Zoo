from flask import Flask, render_template
import sys
import os

# Add the project root to the Python path to allow importing modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

app = Flask(__name__, template_folder='.') # Set template_folder to current directory

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # To run: navigate to the project root in terminal and run `python -m webapp.app`
    # Or from the webapp directory: `python app.py`
    # Ensure requirements are installed: `pip install -r requirements.txt`
    app.run(debug=True, host='0.0.0.0', port=5001) # Use a different port than the API