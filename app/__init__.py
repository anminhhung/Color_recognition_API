from flask import Flask
import os 

from app.config import BaseConfig
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

app.config.from_object(BaseConfig)
db = SQLAlchemy(app)

from app.tracking import tracker
app.register_blueprint(tracker)

from app.models import *