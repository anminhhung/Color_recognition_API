# vim: expandtab:ts=4:sw=4
import cv2
from shapely.geometry import Point, Polygon
import numpy as np 
from utils.parser import get_config
import os

# setup config
cfg = get_config()
cfg.merge_from_file('configs/service.yaml')

VEHICLE_IMAGE = cfg.SERVICE.VEHICLE_IMAGE
STORE_FRAME = cfg.SERVICE.STORE_FRAME

COLOR_LIST = [(255,0,255), (255,100,0), (0,255,0), (139, 69, 19), (132, 112, 255), (0, 154, 205), (0, 255, 127), (238, 180, 180),
                  (0, 100, 0), (238, 106, 167), (221, 160, 221), (0, 128, 128)]

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # flag delete
        self.flag_delete = False

        # attribute
        self.vehicle_color = 'Unknown'
        self.vehicle_type = 'Unknown'

        # add bbox, score, class_name
        self.bbox = None
        self.score = 0.0
        self.class_name = None

        # number_centroid for each moi
        self.list_moi = []

        # moi 
        self.moi = None

        # point_in and point_out
        self.point_in = None 
        self.point_out = None 

        # frame_in and frame_out
        self.frame_in = None
        self.frame_out = None

        # vehicle_path (store image has best score)
        # path_image = vehicle/<cam>/<class>/frame_<cnt_frame>.jpg
        self.path_image = None

        # bbox in frame
        self.frame_path = None

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        # flag recog attribute
        self.flag_attribute = False

        self._n_init = n_init
        self._max_age = max_age
        self.track_line = []

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def check_in_polygon(self, center_point, polygon):
        pts = Point(center_point[0], center_point[1])
        if polygon.contains(pts):
            return True
        
        return False

    def update(self, kf, detection, roi_split_region, cnt_frame, cam_name):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)
        
        # add bounding box, class name, score
        self.bbox = detection.to_tlbr()
        print("vehicle_box: ", self.bbox)
        if detection.confidence >= self.score:
            self.score = detection.confidence
            self.class_name = detection.cls
            self.flag_attribute = True
            self.path_image = os.path.join(VEHICLE_IMAGE, cam_name, self.class_name , 'id_{}_frame_{}.jpg'.format(self.track_id, cnt_frame))
            # crop and save image
            frame_path = os.path.join(STORE_FRAME, cam_name, "frame_" + str(cnt_frame) +  ".jpg")
            frame = cv2.imread(frame_path)
            crop_image = frame[int(self.bbox[1]):int(self.bbox[1])+(int(self.bbox[3]-int(self.bbox[1]))), \
                                int(self.bbox[0]):int(self.bbox[0])+(int(self.bbox[2]-int(self.bbox[0])))]
            cv2.imwrite(self.path_image, crop_image)
        else:
            # set flag attribute     
            self.flag_attribute = False
            
        # frame path 
        self.frame_path = os.path.join(STORE_FRAME, cam_name, 'frame_{}.jpg'.format(cnt_frame))

        x,y,w,h = self.to_tlwh()
        center_x = int(x+w/2)
        center_y = int(y+h/2)
        centroid = (center_x, center_y)

        # check point in 
        if self.point_in == None:
            self.point_in = centroid
        
        self.point_out = centroid

        self.track_line.append(centroid)

        # check frame in 
        if self.frame_in == None:
            self.frame_in = cnt_frame
        
        self.frame_out = cnt_frame

        # check moi
        if len(self.list_moi) == 0:
            self.list_moi = [0] * len(roi_split_region)

        cnt = 0
        while cnt < len(roi_split_region):
            polygon_ROI = roi_split_region[cnt]
            polygon_ROI = Polygon(polygon_ROI)
            if self.check_in_polygon(centroid, polygon_ROI):
                self.list_moi[cnt] += 1

            cnt += 1
        
        distribution = np.array(self.list_moi) / len(self.list_moi)
        # print("distribution: ", distribution)
        # classify moi
        self.moi = np.argmax(distribution) + 1 # b/c list number moi: 1,2,3,4,....,n
        # print("moi: ", self.moi)
        # print("number voting for each moi: ", self.list_moi)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        
        
    def mark_missed(self, image):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
            # visualize before remove
            try:
                cv2.circle(image, (int(self.point_out[0]), int(self.point_out[1])), 12, COLOR_LIST[self.moi - 1], -1)
                self.flag_delete = True
            except:
                pass
        elif self.time_since_update > self._max_age:
            # visualize before remove
            try:
                cv2.circle(image, (int(self.point_out[0]), int(self.point_out[1])), 12, COLOR_LIST[self.moi - 1], -1)
                self.flag_delete = True
            except:
                pass

            self.state = TrackState.Deleted
        

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""

        return self.state == TrackState.Deleted

    def draw_track_line(self,image):
        if len(self.track_line) > 1:
          for i in range(len(self.track_line)-1):
            p1 = self.track_line[i+1]
            p2 = self.track_line[i]
            image = cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), (255,255,0), 3)
        return image