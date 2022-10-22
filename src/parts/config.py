from enum import Enum

SMPL_JOINT_22 = {
    "Pelvis": 0,
    "LHip": 1,
    "RHip": 2,
    "Spine1": 3,
    "LKnee": 4,
    "RKnee": 5,
    "Spine2": 6,
    "LAnkle": 7,
    "RAnkle": 8,
    "Spine3": 9,
    "LFoot": 10,
    "RFoot": 11,
    "Head": 12,
    "LCollar": 13,
    "RCollar": 14,
    "Nose": 15,
    "LShoulder": 16,
    "RShoulder": 17,
    "LElbow": 18,
    "RElbow": 19,
    "LWrist": 20,
    "RWrist": 21,
}


class DirName(Enum):
    MDM = "01_MDM"
    PERSONAL = "02_personal"
