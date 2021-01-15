from flask import Flask
import os 

from app.config import BaseConfig
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, template_folder="templates", static_folder="static")

app.config.from_object(BaseConfig)
db = SQLAlchemy(app)

from app.tracking import tracker
from app.main import main

app.register_blueprint(tracker)
app.register_blueprint(main)

from app.models import *