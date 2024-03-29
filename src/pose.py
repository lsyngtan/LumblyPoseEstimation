import numpy as np

PART_NAMES_COCO = [
    'nose', 
    'neck',  
    'rshoulder', 
    'relbow', 
    'rwrist',
    'lshoulder',
    'lelbow',
    'lwrist', 
    'rhip', 
    'rknee',
    'rankle',
    'lhip',
    'lknee',
    'lankle',
    'reye',
    'leye',
    'rear',
    'lear'
]

PART_NAMES_BODY_25 = [
    "Nose",*
    "Neck",*
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "MidHip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",*
    "LEye",*
    "REar",*
    "LEar",*
    "LBigToe",
    "LSmallToe",
    "LHeel",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "Background"
]
class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)

        for pose in self.poses:
            for attr, part in pose:
                setattr(pose, attr, np.array(part) / mean_torso)

class Pose:

    PART_NAMES = [
        "nose",
        "neck",
        "rshoulder",
        "relbow",
        "rwrist",
        "lshoulder",
        "lelbow",
        "lwrist",
        "midhip",
        "rhip",
        "rknee",
        "rankle",
        "lhip",
        "lknee",
        "lankle",
        "reye",
        "leye",
        "rear",
        "lear",
        "lbigtoe",
        "lsmalltoe",
        "lheel",
        "rbigtoe",
        "rsmalltoe",
        "rheel",
        "background"
    ]

    def __init__(self, parts, model = 'BODY_25'):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 18 * 3 (COCO) or 25 * 3 (BODY_25) ndarray of x, y, confidence values
        """
        # if model == 'BODY_25':
        #     self.PART_NAMES = [part.lower() for part in PART_NAMES_BODY_25]
        # else: 
        #     self.PART_NAMES = PART_NAMES_COCO

        for name, vals in zip(self.PART_NAMES, parts):
            setattr(self, name, Part(vals))
    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out
    
    def print(self, parts):
        out = ""
        for name in parts:
            if not name in self.PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out

class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = self.c != 0.0

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))