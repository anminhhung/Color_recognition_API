import datetime
from flask_sqlalchemy import SQLAlchemy
from app import db

class BaseModel(db.Model):
    """Base data model for all objects"""
    __abstract__ = True

    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        """Define a base way to print models"""
        return '%s(%s)' % (self.__class__.__name__, {
            column: value
            for column, value in self._to_dict().items()
        })

    def json(self):
        """
                Define a base way to jsonify models, dealing with datetime objects
        """
        return {
            column: value if not isinstance(value, datetime.date) else value.strftime('%Y-%m-%d')
            for column, value in self._to_dict().items()
        }

class Camera(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of camera
    cam_name = db.Column(db.String(10), index=True) # name of camera
    info_cam = db.relationship('InfoCam', backref='infocam', lazy='dynamic')
    info_moi = db.relationship('Moi', backref='infomoi', lazy='dynamic')
    info_traffic = db.relationship('Traffic', backref='infotraffic', lazy='dynamic')

    def __init__(Self, cam_name):
        self.cam_name = cam_name
    
    def __repr__(self):
        return '<Camera name: {}>'.format(self.cam_name)

class InfoCam(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of info
    sum_vehicle = db.Column(db.Integer, index=True) # total of vehicles
    sum_xe_may = db.Column(db.Integer, index=True) # total of xe may
    sum_ba_gac = db.Column(db.Integer, index=True) # total of xe ba gac
    sum_taxi = db.Column(db.Integer, index=True) # total of taxi
    sum_car = db.Column(db.Integer, index=True) # total of car
    sum_ban_tai = db.Column(db.Integer, index=True) # total of xe ban tai
    sum_cuu_thuong = db.Column(db.Integer, index=True) # total of xe cuu thuong
    sum_xe_khach = db.Column(db.Integer, index=True) # total of xe khach
    sum_bus = db.Column(db.Integer, index=True) # total of xe bus
    sum_tai = db.Column(db.Integer, index=True) # total of xe tai
    sum_container = db.Column(db.Integer, index=True) # total of container
    cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), index=True)
    
    def __init__(self, vehicles, xemay, bagac, taxi, car, bantai, cuuthuong, xekhach, bus, tai, container):
        self.sum_vehicle = vehicles
        self.sum_xe_may = xemay
        self.sum_ba_gac = bagac
        self.sum_taxi = taxi 
        self.sum_car = car
        self.sum_ban_tai = bantai 
        self.sum_cuu_thuong = cuuthuong
        self.sum_xe_khach = xekhach
        self.sum_bus = bus
        self.sum_tai = tai
        self.sum_container = container

    def __repr__(self):
        return '<Number of vehicles: {}>'.format(self.sum_vehicle)

class Moi(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of MOI
    moi_name = db.Column(db.String(30), index=True) # name of MOI 
    cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), index=True)
    info_moi = db.relationship('InfoMoi', backref='infomoi', lazy='dynamic')

    def __init__(self, moi_name):
        self.moi_name = moi_name

    def __repr__(self):
        return '<Name of Moi: {}>'.format(self.moi_name)

class InfoMoi(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of info
    sum_vehicle = db.Column(db.Integer, index=True) # total of vehicles
    sum_xe_may = db.Column(db.Integer, index=True) # total of xe may
    sum_ba_gac = db.Column(db.Integer, index=True) # total of xe ba gac
    sum_taxi = db.Column(db.Integer, index=True) # total of taxi
    sum_car = db.Column(db.Integer, index=True) # total of car
    sum_ban_tai = db.Column(db.Integer, index=True) # total of xe ban tai
    sum_cuu_thuong = db.Column(db.Integer, index=True) # total of xe cuu thuong
    sum_xe_khach = db.Column(db.Integer, index=True) # total of xe khach
    sum_bus = db.Column(db.Integer, index=True) # total of xe bus
    sum_tai = db.Column(db.Integer, index=True) # total of xe tai
    sum_container = db.Column(db.Integer, index=True) # total of container
    moi_id = db.Column(db.Integer, db.ForeignKey('moi.id'), index=True)

    def __init__(self, vehicles, xemay, bagac, taxi, car, bantai, cuuthuong, xekhach, bus, tai, container):
        self.sum_vehicle = vehicles
        self.sum_xe_may = xemay
        self.sum_ba_gac = bagac
        self.sum_taxi = taxi 
        self.sum_car = car
        self.sum_ban_tai = bantai 
        self.sum_cuu_thuong = cuuthuong
        self.sum_xe_khach = xekhach
        self.sum_bus = bus
        self.sum_tai = tai
        self.sum_container = container
    
    def __repr__(self):
        return '<Number of vehicles: {}>'.format(self.sum_vehicle)

class Traffic(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of traffic
    point_in = db.Column(db.String(25), index=True) # point in MOI 
    point_out = db.Column(db.String(25), index=True) # point out MOI
    frame_in = db.Column(db.Integer, index=True) # frame in MOI 
    frame_out = db.Column(db.Integer, index=True) # frame out MOI 
    cam_id = db.Column(db.Integer, db.ForeignKey('camera.id'), index=True)
    info_vehicles = db.relationship('Vehicles', backref='infovehicles', lazy='dynamic')

    def __init__(self, point_in, point_out, frame_in, frame_out):
        self.point_in = point_in
        self.point_out = point_out
        self.frame_in = frame_in
        self.frame_out = frame_out
    
    def __repr__(self):
        return '<Traffic table>'

class Vehicles(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of vehicle
    vehicle_name = db.Column(db.String(10), index=True) # name of vehicle
    vehicle_score = db.Column(db.Float, index=True) # confidence of vehicle
    vehicle_path = db.Column(db.String(100), index=True) # path of Vehicle cropped image 
    number_track = db.Column(db.Integer, index=True) # track_id of vehicle
    traffic_id = db.Column(db.Integer, db.ForeignKey('traffic.id'), index=True)
    info_frame = db.relationship('Frames', backref='infoframes', lazy='dynamic')
    info_type = db.relationship('Type', backref='infotype', lazy='dynamic')
    info_color = db.relationship('Color', backref='infocolor', lazy='dynamic')

    def __init__(self, vehicle_name, vehicle_score, vehicle_path, number_track):
        self.vehicle_name = vehicle_name
        self.vehicle_score = vehicle_score
        self.vehicle_path = vehicle_path
        self.number_track = number_track
    
    def __repr__(self):
        return '<Name of vehicles: {}>'.format(self.vehicle_name)

class Frames(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of frame
    frame_path = db.Column(db.String(100), index=True) # path of frame
    bbox = db.Column(db.String(25), index=True) # bounding box 
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), index=True)

    def __init__(self, frame_path, bbox):
        self.frame_path = frame_path
        self.bbox = bbox
    
    def __repr__(self):
        return '<Frame: {}>'.format(self.frame_path)

class Type(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of type
    vehicle_type = db.Column(db.String(10), index=True) # type of vehicles
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), index=True)

    def __init__(self, vehicle_type):
        self.vehicle_type = vehicle_type
    
    def __repr__(self):
        return '<Vehicle type: {}>'.format(self.vehicle_type)

class Color(BaseModel, db.Model):
    id = db.Column(db.Integer, primary_key=True) # id of Color 
    vehicle_color = db.Column(db.String(20), index=True) # color of vehicles
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicles.id'), index=True)

    def __init__(self, vehicle_color):
        self.vehicle_color = vehicle_color
    
    def __repr__(self):
        return '<vehicle color: {}>'.format(self.vehicle_color)