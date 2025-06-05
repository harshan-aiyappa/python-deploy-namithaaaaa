# freezer.py
from flask_frozen import Freezer
from main import main  # your Flask app instance

freezer = Freezer(main)

if __name__ == '__main__':
    freezer.freeze()
