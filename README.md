create venv 
python -m venv venv
activate venv
source venv/bin/activate
install requirements by 
pip install -r requirements.txt

run ML model to dowlaod weights

python vgg_model.py

run flask app

flask run




