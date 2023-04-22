import pickle

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import os

from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from scipy.stats import zscore
from scipy.signal import find_peaks
from scipy import ndimage

import libfmp.b
import libfmp.c4
import libfmp.c7

import cv2
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# create function to get pkl file from blob storage
def get_pkl(blob_name, container_name):
    account_name = os.environ["AccountName"]
    account_key = os.environ["AccountKey"]
    #create a client to interact with blob storage
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=' + account_name + ';AccountKey=' + account_key + ';EndpointSuffix=core.windows.net'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    #use the client to connect to the container
    container_client = blob_service_client.get_container_client(container_name)

    # try finding blob with .pkl or .PKL extension
    temp_blob_name = blob_name + '.pkl'
    blob_client = container_client.get_blob_client(temp_blob_name)
    if not blob_client.exists():
      return ["blob does not exist", 404]
    blob_name = temp_blob_name
    # generate a StorageStreamDownloader object
    blob_stream = blob_client.download_blob(0)
    # get the blob content
    blob_content = blob_stream.readall()      

    return blob_content

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

PART_NAMES = [
    "nose",
    "leye",
    "reye",
    "lear",
    "rear",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhip",
    "rhip",
    "lknee",
    "rknee",
    "lankle",
    "rankle",
    "neck"
] 

class PoseSequence:
  def __init__(self, sequence):
      self.poses = []
      for parts in sequence:
          pose_obj = Pose(parts)
          pose_obj
          self.poses.append(pose_obj)
      
      # extract 
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
      "leye",
      "reye",
      "lear",
      "rear",
      "lshoulder",
      "rshoulder",
      "lelbow",
      "relbow",
      "lwrist",
      "rwrist",
      "lhip",
      "rhip",
      "lknee",
      "rknee",
      "lankle",
      "rankle",
      "neck"
    ] 

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 18 * 3 (COCO) or 25 * 3 (BODY_25) ndarray of x, y, confidence values
        """
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

def _bird_dog(pose_seq, pkl_path = None):
    # find the arm and leg that is currently used
    poses = pose_seq.poses

    joints_right = [{
                        "neck" : pose.neck,
                        "shoulder" : pose.rshoulder, 
                        "elbow" : pose.relbow, 
                        "wrist" : pose.rwrist, 
                        "hip" : pose.rhip, 
                        "knee" : pose.rknee, 
                        "ankle" : pose.rankle 
                    } for pose in poses]
    joints_left = [{
                        "neck" : pose.neck,
                        "shoulder" : pose.lshoulder, 
                        "elbow" : pose.lelbow, 
                        "wrist" : pose.lwrist, 
                        "hip" : pose.lhip, 
                        "knee" : pose.lknee, 
                        "ankle" : pose.lankle
                    } for pose in poses]


    # filter out data points where a part does not exist
    joints_right = [joint for joint in joints_right if all(joint[part].exists for part in joint)]
    joints_left = [joint for joint in joints_left if all(joint[part].exists for part in joint)]

    # elbow to shoulder
    right_upper_arm_vecs = np.array([(joint["shoulder"].x - joint["elbow"].x, joint["shoulder"].y - joint["elbow"].y) for joint in joints_right])
    # hip to shoulder
    right_torso_vecs = np.array([(joint["shoulder"].x - joint["hip"].x, joint["shoulder"].y - joint["hip"].y) for joint in joints_right])
    # wrist to elbow
    right_forearm_vecs = np.array([(joint["elbow"].x - joint["wrist"].x, joint["elbow"].y - joint["wrist"].y) for joint in joints_right])
    # hip to knee
    right_thigh_vecs = np.array([(joint["hip"].x - joint["knee"].x, joint["hip"].y - joint["knee"].y) for joint in joints_right])
    # ankle to knee
    right_shank_vecs = np.array([(joint["knee"].x - joint["ankle"].x, joint["knee"].y - joint["ankle"].y) for joint in joints_right])
    
    # elbow to shoulder
    left_upper_arm_vecs = np.array([(joint["shoulder"].x - joint["elbow"].x, joint["shoulder"].y - joint["elbow"].y) for joint in joints_left])
    # hip to shoulder
    left_torso_vecs = np.array([(joint["shoulder"].x - joint["hip"].x, joint["shoulder"].y - joint["hip"].y) for joint in joints_left])
    # wrist to elbow
    left_forearm_vecs = np.array([(joint["elbow"].x - joint["wrist"].x, joint["elbow"].y - joint["wrist"].y) for joint in joints_left])
    # hip to knee
    left_thigh_vecs = np.array([(joint["hip"].x - joint["knee"].x, joint["hip"].y - joint["knee"].y) for joint in joints_left])
    # ankle to knee
    left_shank_vecs = np.array([(joint["knee"].x - joint["ankle"].x, joint["knee"].y - joint["ankle"].y) for joint in joints_left])

    # normalize vectors
    right_upper_arm_vecs = right_upper_arm_vecs / np.expand_dims(np.linalg.norm(right_upper_arm_vecs, axis=1), axis=1)
    right_torso_vecs = right_torso_vecs / np.expand_dims(np.linalg.norm(right_torso_vecs, axis=1), axis=1)
    right_forearm_vecs = right_forearm_vecs / np.expand_dims(np.linalg.norm(right_forearm_vecs, axis=1), axis=1)
    right_thigh_vecs = right_thigh_vecs / np.expand_dims(np.linalg.norm(right_thigh_vecs, axis=1), axis=1)
    right_shank_vecs = right_shank_vecs / np.expand_dims(np.linalg.norm(right_shank_vecs, axis=1), axis=1)

    left_upper_arm_vecs = left_upper_arm_vecs / np.expand_dims(np.linalg.norm(left_upper_arm_vecs, axis=1), axis=1)
    left_torso_vecs = left_torso_vecs / np.expand_dims(np.linalg.norm(left_torso_vecs, axis=1), axis=1)
    left_forearm_vecs = left_forearm_vecs / np.expand_dims(np.linalg.norm(left_forearm_vecs, axis=1), axis=1)
    left_thigh_vecs = left_thigh_vecs / np.expand_dims(np.linalg.norm(left_thigh_vecs, axis=1), axis=1)
    left_shank_vecs = left_shank_vecs / np.expand_dims(np.linalg.norm(left_shank_vecs, axis=1), axis=1)

    vector_dict = {       
        "right_upper_arm_vecs": right_upper_arm_vecs,
        "right_torso_vecs": right_torso_vecs,
        "right_forearm_vecs": right_forearm_vecs,
        "right_thigh_vecs": right_thigh_vecs, 
        "right_shank_vecs": right_shank_vecs,

        "left_upper_arm_vecs": left_upper_arm_vecs,
        "left_torso_vecs": left_torso_vecs, 
        "left_forearm_vecs": left_forearm_vecs,
        "left_thigh_vecs": left_thigh_vecs,
        "left_shank_vecs": left_shank_vecs  
    }


     # calculate angles between body parts
    right_upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_upper_arm_vecs, right_torso_vecs), axis=1), -1.0, 1.0)))
    right_upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_upper_arm_vecs, right_forearm_vecs), axis=1), -1.0, 1.0)))
    right_torso_thigh_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_torso_vecs,  right_thigh_vecs), axis=1), -1.0, 1.0)))
    right_thigh_shank_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_thigh_vecs, right_shank_vecs), axis=1), -1.0, 1.0)))
    
    left_upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_upper_arm_vecs, left_torso_vecs), axis=1), -1.0, 1.0)))
    left_upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_upper_arm_vecs, left_forearm_vecs), axis=1), -1.0, 1.0)))
    left_torso_thigh_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_torso_vecs, left_thigh_vecs), axis=1), -1.0, 1.0)))
    left_thigh_shank_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_thigh_vecs, left_shank_vecs), axis=1), -1.0, 1.0)))

    angle_dict = {       
        "right_upper_arm_torso_angles" : right_upper_arm_torso_angles,
        "right_upper_arm_forearm_angles" : right_upper_arm_forearm_angles,
        "right_torso_thigh_angles" : right_torso_thigh_angles,
        "right_thigh_shank_angles" : right_thigh_shank_angles,        
        "left_upper_arm_torso_angles" : left_upper_arm_torso_angles,
        "left_upper_arm_forearm_angles" : left_upper_arm_forearm_angles,
        "left_torso_thigh_angles" : left_torso_thigh_angles,
        "left_thigh_shank_angles" : left_thigh_shank_angles        
    }

    # Perform median filtering
    filtered_dict = {}
    for key in angle_dict.keys():
        filtered_dict[key] = medfilt(medfilt(angle_dict[key], 5), 5)

    # Save relevant dictionaries
    if pkl_path is not None: 
      with open(f"{pkl_path}_vectors.pkl", 'wb') as f:
        pickle.dump(vector_dict, f)
    if pkl_path is not None: 
      with open(f"{pkl_path}_angles.pkl", 'wb') as f:
        pickle.dump(angle_dict, f)
    if pkl_path is not None: 
      with open(f"{pkl_path}_filtered.pkl", 'wb') as f:
        pickle.dump(filtered_dict, f)

    return vector_dict, angle_dict, filtered_dict

def preprocess_pose(all_keypoints):
  # Calculate neck as average b/w L and R shoulders
  all_keypoints = np.append(
      all_keypoints, 
      ((all_keypoints[:, 5, :] + all_keypoints[:, 6, :])/2).reshape(-1, 1, 3), 
      axis = 1
  )
  # Convert to PoseSequence object and extract unit vectors + angles
  pose_seq = PoseSequence(all_keypoints)
  vector_dict, _, filtered_dict = _bird_dog(pose_seq)

  return vector_dict, all_keypoints, filtered_dict

# Eqn 1.
def get_FR(feature):
    return np.sum(np.abs(np.gradient(feature)))

def get_FRS(exemplar_vectors):
    FRS = {}
    for key in exemplar_vectors.keys():
      if 'thigh' in key.lower():
        for coord in [0, 1]:
            feat_ranking = get_FR(exemplar_vectors[key][:, coord])
            FRS[f"{key}_{coord}"] = {'FR': feat_ranking, 'TCT': 0}
    
    
    return pd.DataFrame(FRS).T

# Cubic interpolation + derivative
def get_DTW_spline(feature):
    x = range(len(feature))
    cubic_spline = CubicSpline(x, feature)
    
    return x, cubic_spline(x), np.gradient(cubic_spline(x))

def get_top_ranking_motion_ft(FRS, vector):
  key = FRS.index[0][:-2]
  coord = int(FRS.index[0][-1])
  print(key, coord)
  return key, vector[key][:, coord]

def get_metrics(idxs, gt_bounds_alt, TE):
  # Add a time error to each GT
  te_rep_bounds = []
  for i in range(1, len(gt_bounds_alt)):
    te_rep_bounds.append((
        gt_bounds_alt[i-1] - TE,
        gt_bounds_alt[i] + TE
    ))
  # Now compare each GT segment w/ that of the algorithmic segmentations
  comparison_list = []
  for bound in te_rep_bounds:
    contained_segments =  [i for i in idxs if i[0] >= bound[0] and i[1] <= bound[1]]
    comparison_list.append(contained_segments)
    print(bound, contained_segments)
  tp = len([i for i in comparison_list if len(i) > 0])
  fp = len([i for i in comparison_list if len(i) > 1])
  fn = len([i for i in comparison_list if len(i) == 0])

  return {"TP": tp, "FP": fp, "FN": fn}

def compute_matching_function_dtw(X, Y, stepsize=2):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X (np.ndarray): Query feature sequence (given as K x N matrix)
        Y (np.ndarray): Database feature sequence (given as K x M matrix)
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        Delta (np.ndarray): DTW-based matching function
        C (np.ndarray): Cost matrix
        D (np.ndarray): Accumulated cost matrix
    """
    C = libfmp.c7.cost_matrix_dot(X, Y)
    if stepsize == 1:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
    return Delta, C, D

def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        pos (np.ndarray): End positions of matches
        D (np.ndarray): Accumulated cost matrix
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches

def perform_subseq_dtw_alt(exemplar_spline, query_spline, num = 10):
  Delta, C, D = compute_matching_function_dtw(
        zscore(exemplar_spline).reshape(1, -1), 
        zscore(query_spline).reshape(1, -1),
        stepsize=2
    )
  pos = libfmp.c7.mininma_from_matching_function(Delta, rho=2*len(exemplar_spline)//3, tau=0.1, num=num)
  matches = matches_dtw(pos, D, stepsize=2)

  return matches

def rename_angle_columns(rep_df, moving_limbs_dict):
  rename_dict = {}
  for col in rep_df.columns:
    col_suffix = '_'.join(col.split('_')[1:])
    # Rename arm columns
    if "arm" in col.lower() and moving_limbs_dict['moving_arm'] in col.lower():
      rename_dict[col] = f"moving_{col_suffix}"
    elif "arm" in col.lower() and moving_limbs_dict['moving_arm'] not in col.lower():
      rename_dict[col] = f"stationary_{col_suffix}"

    # Rename leg columns
    elif ("shank" in col.lower() or "thigh" in col.lower()) and moving_limbs_dict['moving_leg'] in col.lower():
      rename_dict[col] = f"moving_{col_suffix}"
    elif ("shank" in col.lower() or "thigh" in col.lower()) and moving_limbs_dict['moving_leg'] not in col.lower():
      rename_dict[col] = f"stationary_{col_suffix}"
  return rename_dict

def get_total_rom(motion_data):
  return np.sum(np.abs(np.gradient(motion_data)))

def get_total_moving_side_angles(filtered_angle_dict):
  sides = ['right', 'left']
  right_arm_angles = filtered_angle_dict['right_upper_arm_torso_angles']
  left_arm_angles = filtered_angle_dict['left_upper_arm_torso_angles']
  right_arm_rom, left_arm_rom = get_total_rom(right_arm_angles), get_total_rom(left_arm_angles)
  moving_arm = np.argmax([right_arm_rom, left_arm_rom])

  right_leg_angles = filtered_angle_dict['right_torso_thigh_angles']
  left_leg_angles = filtered_angle_dict['left_torso_thigh_angles']
  right_leg_rom, left_leg_rom = get_total_rom(right_leg_angles), get_total_rom(left_leg_angles)
  moving_leg = np.argmax([right_leg_rom, left_leg_rom])

  moving_limb_dict = {
      "moving_arm": sides[moving_arm], 
      "moving_leg": sides[moving_leg]
  }
  print(moving_limb_dict) 

  return moving_limb_dict 

def get_total_moving_side_vectors(query_vectors):
  sides = ['right', 'left']
  right_arm_angles = query_vectors['right_forearm_vecs'][:, 1]
  left_arm_angles = query_vectors['left_forearm_vecs'][:, 1]
  right_arm_rom, left_arm_rom = get_total_rom(right_arm_angles), get_total_rom(left_arm_angles)
  moving_arm = np.argmax([right_arm_rom, left_arm_rom])

  right_leg_angles = query_vectors['right_thigh_vecs'][:, 1]
  left_leg_angles = query_vectors['left_thigh_vecs'][:, 1]
  right_leg_rom, left_leg_rom = get_total_rom(right_leg_angles), get_total_rom(left_leg_angles)
  moving_leg = np.argmax([right_leg_rom, left_leg_rom])

  moving_limb_dict = {
      "moving_arm": sides[moving_arm], 
      "moving_leg": sides[moving_leg]
  }
  print(moving_limb_dict) 

  return moving_limb_dict 


def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    # kpts_above_thresh_absolute = kpts_absolute_xy[
    #     kpts_scores > keypoint_threshold, :]
    # keypoints_all.append(kpts_above_thresh_absolute)
    keypoints_all.append(kpts_absolute_xy)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

def get_head_orientation(img_data, query_keypoints):
  query_pt = int(len(img_data)/2)
  height, width, channel = img_data[query_pt].shape

  # Extract human interpretable coordinates
  query_keypoints = np.expand_dims(
      np.expand_dims(
          query_keypoints[query_pt, :17, :], 
          axis = 0
      ), 
      axis = 0
  )
  keypoint_locs, _ ,_ = _keypoints_and_edges_for_display(
      query_keypoints, height, width
  )
  keypoint_interp_dict = dict(zip(PART_NAMES, keypoint_locs))
  if 'rankle' in keypoint_interp_dict.keys() and 'nose' in keypoint_interp_dict.keys():
    if keypoint_interp_dict['nose'][0] > keypoint_interp_dict['rankle'][0]:
      return "right", keypoint_locs
    else:
      return "left", keypoint_locs
  else:
    return "Not all keypoints detected", keypoint_locs

# CHANGE BASED ON WHERE VIDEO PKLS ARE STORED
def load_exemplar(video_path, container_name):
  # Load keypoints from .pkl path
  exemplar_keypoints = pickle.loads(get_pkl(video_path, container_name))

  # Calculate neck as average b/w L and R shoulders
  exemplar_keypoints = np.append(
      exemplar_keypoints, 
      ((exemplar_keypoints[:, 5, :] + exemplar_keypoints[:, 6, :])/2).reshape(-1, 1, 3), 
      axis = 1
  )
  # Convert to PoseSequence object and extract unit vectors + angles
  pose_seq = PoseSequence(exemplar_keypoints)
  vector_dict, _, filtered_dict = _bird_dog(pose_seq)

  return vector_dict, exemplar_keypoints, filtered_dict

def perform_segmentation_updated(raw_keypoints, raw_img_data, container_name, exercise_set, exercise_name):
  """
  Segments repetitions using sDTW 
  """

  print(raw_img_data.shape)
  # Preprocess USER video
  query_vectors, query_keypoints, filtered_angle_dict = preprocess_pose(raw_keypoints)

  # Detect moving side
  moving_limb_dict = get_total_moving_side(filtered_angle_dict)

  # Detect head orientation
  head_orientation, interp_keypts =  get_head_orientation(
      raw_img_data, 
      query_keypoints = query_keypoints
  )
  print("head_orientation:", head_orientation)

  # Select appropriate exemplar video
  if moving_limb_dict['moving_arm'] == 'left' and moving_limb_dict['moving_leg'] == 'right' and head_orientation == 'right':
    user_case = 'case_1'
  elif moving_limb_dict['moving_arm'] == 'right' and moving_limb_dict['moving_leg'] == 'left' and head_orientation == 'right':
    user_case = 'case_2'
  elif moving_limb_dict['moving_arm'] == 'right' and moving_limb_dict['moving_leg'] == 'left' and head_orientation == 'left':
    user_case = 'case_3'
  elif moving_limb_dict['moving_arm'] == 'left' and moving_limb_dict['moving_leg'] == 'right' and head_orientation == 'left':
    user_case = 'case_4'
  else: 
    print("Bird Dog form invalid; moving arm and leg are the same")
    return

  # CHANGE DEPENDING ON VIDEO_NAMES
  exemplar_vid_dict = {
      "case_1": "LRR", # moving arm left, moving leg right, head right
      "case_2": "RLR", # moving arm right, moving leg left, head right
      "case_3": "RLL", # moving arm right, moving leg left, head left
      "case_4": "LRL"  # moving arm left, moving leg right, head left
  }
  exemplar_vector, _, _ = load_exemplar("{}/{}/reference_videos/{}".format(exercise_set, exercise_name, exemplar_vid_dict[user_case]), container_name)

  # Calculate total change associated w/ each motion feature and rank
  FRS = get_FRS(exemplar_vector)
  FRS = FRS.sort_values(by = 'FR', ascending = False)

  # Spline fitting
  ft_name, exemplar_feature = get_top_ranking_motion_ft(FRS, exemplar_vector)
  xs, exemplar_spline, dt = get_DTW_spline(exemplar_feature)
  
  ft_name, query_feature = get_top_ranking_motion_ft(FRS, query_vectors)
  xs, query_spline, dt = get_DTW_spline(query_feature)

  # query_spline, exemplar_spline = ndimage.gaussian_filter1d(query_spline, 3), ndimage.gaussian_filter1d(exemplar_spline, 3)

  idxs = perform_subseq_dtw_alt(exemplar_spline, query_spline)
  idxs = sorted(idxs, key=lambda x: x[0])

  return idxs, filtered_angle_dict, query_vectors, query_keypoints, moving_limb_dict
  
def extract_rep_dfs_and_keypts(img_data, angle_dict, vector_dict, 
                               idxs, moving_limbs, query_keypoints):    
  # Check that vectors and angles are aligned
  vector_dict_lengths = [len(vector_dict[key]) for key in vector_dict.keys()]
  angle_dict_lengths = [len(angle_dict[key]) for key in angle_dict.keys()]
  assert np.unique(vector_dict_lengths).shape[0] == 1 and np.unique(angle_dict_lengths).shape[0] == 1 
  assert np.unique(vector_dict_lengths)[0]  == np.unique(angle_dict_lengths)[0]

  data = pd.Series(
    zscore(
        vector_dict[f"{moving_limbs['moving_leg']}_thigh_vecs"][:, 1]
    )
  )
  held_rep_dfs, held_interpret_keypts, held_frame_idxs = [], [], []
  for i in range(len(idxs)):
    # Extract vector data 
    rep_data = data.iloc[idxs[i][0]:idxs[i][1]] 
    
    # Apply Gaussian filter to reduce noise
    smoothed_rep_data = ndimage.gaussian_filter1d(rep_data, 3)
    rep_data = pd.Series(smoothed_rep_data, index = rep_data.index)
    
    # Fit a quadratic to the data and determine whether we are looking for a trough or peak
    model = np.poly1d(
        np.polyfit(rep_data.index, rep_data.values, 2)
    )
    if model.coefficients[0] < 0: # Look for peaks (concave)
      peaks = find_peaks(rep_data)[0]
    else: # Look for a trough (convex)
      peaks = find_peaks(-rep_data)[0]

    # Find the highest peak or lowest trough
    max_peak_idx = abs(rep_data)[rep_data.index[peaks]].idxmax()
    relevant_frame = img_data[max_peak_idx]
    print(max_peak_idx)

    # Get body angles at the held frame
    held_angle_df = pd.DataFrame(angle_dict).loc[max_peak_idx]

    held_angle_df['Rep'] = i
    held_angle_df = pd.DataFrame(held_angle_df).T

    # Rename based on moving/stationary limbs
    rename_dict = rename_angle_columns(held_angle_df, moving_limbs)
    held_angle_df = held_angle_df.rename(columns = rename_dict)

    # Get keypoints at "held" frame 
    processed_keypoints = np.expand_dims(
      np.expand_dims(
          query_keypoints[max_peak_idx, :17, :], 
          axis = 0
      ), 
      axis = 0
    )
    height, width, channel = relevant_frame.shape  
    keypoint_locs, _ ,_ = _keypoints_and_edges_for_display(
        processed_keypoints, 
        height, 
        width
    )
    # Reject reps where we don't see all keypoints (artifacts during start and end of recordings)
    if keypoint_locs.shape[0] < 17:
      continue
    
    held_rep_dfs.append(held_angle_df)
    held_interpret_keypts.append(keypoint_locs)
    held_frame_idxs.append(max_peak_idx)

  return pd.concat(held_rep_dfs), held_interpret_keypts, held_frame_idxs

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def get_held_reps(raw_keypoints, raw_img_data, show_plots = False, prominence = None): 
  # Preprocess USER video
  query_vectors, query_keypoints, filtered_angle_dict = preprocess_pose(raw_keypoints)

  # Detect moving side
  moving_limb_dict = get_total_moving_side_angles(filtered_angle_dict)
  # moving_limb_dict = get_total_moving_side_vectors(query_vectors)

  if moving_limb_dict['moving_leg']  == moving_limb_dict['moving_arm']:
    print("Bird Dog form invalid; moving arm and leg are the same")
    return

  # Detect head orientation
  head_orientation, interp_keypts =  get_head_orientation(
      raw_img_data, 
      query_keypoints = query_keypoints
  )
  print("head_orientation:", head_orientation)
  if head_orientation == "Not all keypoints detected":
    return

  # Check that vectors and angles are aligned
  vector_dict_lengths = [len(query_vectors[key]) for key in query_vectors.keys()]
  angle_dict_lengths = [len(filtered_angle_dict[key]) for key in filtered_angle_dict.keys()]
  assert np.unique(vector_dict_lengths).shape[0] == 1 and np.unique(angle_dict_lengths).shape[0] == 1 
  assert np.unique(vector_dict_lengths)[0]  == np.unique(angle_dict_lengths)[0]

  data = pd.Series(
    zscore(
        query_vectors[f"{moving_limb_dict['moving_leg']}_thigh_vecs"][:, 1]
    )
  )
  # Apply Gaussian filter to reduce noise prior to peak finding
  smoothed_rep_data = ndimage.gaussian_filter1d(data, 3)
  rep_data = pd.Series(smoothed_rep_data, index = data.index)

  # Peak-finding
  if head_orientation == 'left': # convex, therefore look for a trough
    peaks = find_peaks(-rep_data, distance = 30, prominence = prominence, height = prominence)[0]
    max_peak_idxs = rep_data[rep_data.index[peaks]].nsmallest(10).index.to_list()
  elif head_orientation == 'right': # concave, therefore look for a peak
    peaks = find_peaks(rep_data, distance = 30, prominence = prominence, height = prominence)[0]
    max_peak_idxs = rep_data[rep_data.index[peaks]].nlargest(10).index.to_list()
  max_peak_idxs = sorted(max_peak_idxs)
  relevant_frames = raw_img_data[max_peak_idxs]

  held_rep_dfs, held_interpret_keypts = [], []
  for i, relevant_frame in enumerate(relevant_frames):
    idx = max_peak_idxs[i]
    # Extract vector data 
    rep_data = data.iloc[idx] 

    # Get body angles at the held frame
    held_angle_df = pd.DataFrame(filtered_angle_dict).loc[idx]

    held_angle_df['Rep'] = i
    held_angle_df = pd.DataFrame(held_angle_df).T

    # Rename based on moving/stationary limbs
    rename_dict = rename_angle_columns(held_angle_df, moving_limb_dict)
    held_angle_df = held_angle_df.rename(columns = rename_dict)

    # Get keypoints at "held" frame 
    processed_keypoints = np.expand_dims(
      np.expand_dims(
          query_keypoints[idx, :17, :], 
          axis = 0
      ), 
      axis = 0
    )
    height, width, channel = relevant_frame.shape  
    keypoint_locs, _ ,_ = _keypoints_and_edges_for_display(
        processed_keypoints, 
        height, 
        width
    )
    
    if show_plots:
      image_plot = draw_prediction_on_image(relevant_frame, processed_keypoints)
      plt.figure()
      plt.imshow(image_plot)
      plt.title(f"Rep {i}")
    # # Reject reps where we don't see all keypoints (artifacts during start and end of recordings)
    # if keypoint_locs.shape[0] < 17:
    #   continue

    held_rep_dfs.append(held_angle_df)
    held_interpret_keypts.append(keypoint_locs)

  return pd.concat(held_rep_dfs), held_interpret_keypts, max_peak_idxs, moving_limb_dict, query_vectors, filtered_angle_dict

# Feature Extaction functions
def get_ankle_shoulder_diff(interpret_keypts, moving_limbs, n_reps):
  shoulder_ankle_diffs = []
  for i in range(n_reps):
    moving_leg = moving_limbs['moving_leg']

    moving_ankle_height = 1200 - interpret_keypts[i][KEYPOINT_DICT[f"{moving_leg}_ankle"]][1]

    shoulder_sides = ['right', 'left']
    right_shoulder_y = 1200 - interpret_keypts[i][KEYPOINT_DICT["right_shoulder"]][1]
    left_shoulder_y = 1200 - interpret_keypts[i][KEYPOINT_DICT["left_shoulder"]][1]
    higher_shoulder_idx = np.argmax([right_shoulder_y, left_shoulder_y])
    higher_shoulder = np.max([right_shoulder_y, left_shoulder_y])

    shoulder_ankle_diffs.append(
      moving_ankle_height - higher_shoulder
    )

  return shoulder_ankle_diffs

def get_wrist_shoulder_diff(interpret_keypts, moving_limbs, n_reps):
  shoulder_wrist_diffs = []
  for i in range(n_reps):
    moving_arm = moving_limbs['moving_arm']
    moving_wrist_height = 1200 - interpret_keypts[i][KEYPOINT_DICT[f"{moving_arm}_wrist"]][1]

    shoulder_sides = ['right', 'left']
    right_shoulder_y = 1200 - interpret_keypts[i][KEYPOINT_DICT["right_shoulder"]][1]
    left_shoulder_y = 1200 - interpret_keypts[i][KEYPOINT_DICT["left_shoulder"]][1]
    higher_shoulder_idx = np.argmax([right_shoulder_y, left_shoulder_y])
    higher_shoulder = np.max([right_shoulder_y, left_shoulder_y])

    shoulder_wrist_diffs.append(
      moving_wrist_height - higher_shoulder
    )
  return shoulder_wrist_diffs

def extract_features(full_held_rep_df, moving_limbs, interpret_keypts):                 
  feature_vector = full_held_rep_df.copy()
  n_reps = int(full_held_rep_df['Rep'].max()) + 1
  # Extract shoulder ankle and shoulder-wrist differences
  shoulder_ankle_diffs = get_ankle_shoulder_diff(
      interpret_keypts = interpret_keypts, 
      moving_limbs = moving_limbs,
      n_reps = n_reps
  )
  shoulder_wrist_diffs = get_wrist_shoulder_diff(
      interpret_keypts = interpret_keypts, 
      moving_limbs = moving_limbs,
      n_reps = n_reps
  )
  feature_vector['shoulder_ankle_diffs'] = shoulder_ankle_diffs
  feature_vector['shoulder_wrist_diffs'] = shoulder_wrist_diffs

  return feature_vector.set_index('Rep')

# Output from MoveNet goes here:
def get_mistakes(movenet_keypts, movenet_imgs, exercise_set, exercise_name, container_name, trim_idxs = 60):
  mistakes = []
  mistake_frame_dict = {}
  error_flag = False

  try:
    # Trim ends of video
    if trim_idxs is not None:
      movenet_keypts = movenet_keypts[trim_idxs:-trim_idxs]
      movenet_imgs = movenet_imgs[trim_idxs:-trim_idxs]

    # Loads the data from keypoints and image frames provided by MoveNet then performs rep segmentation
    # Extracts the held frames, keypoints, and body angles at the held frame for each rep  
    held_rep_df, held_interpret_keypts, held_frames_idxs, moving_limb_dict, query_vectors, filtered_angle_dict = get_held_reps(
        movenet_keypts, 
        movenet_imgs, 
        show_plots = False, 
        prominence = 0.8
    )
    # Extract feature vector
    feature_vector = extract_features(held_rep_df, moving_limb_dict, held_interpret_keypts)

    # CHANGE TO REFLECT WHERE MODELS AND SCALERS ARE STORED
    leg_model = pickle.loads(get_pkl("{}/{}/models/leg_models/model".format(exercise_set, exercise_name), container_name))
    leg_scaler = pickle.loads(get_pkl("{}/{}/models/leg_models/scaler".format(exercise_set, exercise_name), container_name))
    bent_arm_model = pickle.loads(get_pkl("{}/{}/models/bent_arm_models/model".format(exercise_set, exercise_name), container_name))
    bent_arm_scaler = pickle.loads(get_pkl("{}/{}/models/bent_arm_models/scaler".format(exercise_set, exercise_name), container_name))
    arm_model = pickle.loads(get_pkl("{}/{}/models/arm_models/model".format(exercise_set, exercise_name), container_name))
    arm_scaler = pickle.loads(get_pkl("{}/{}/models/arm_models/scaler".format(exercise_set, exercise_name), container_name))
                      
    X_leg = feature_vector[['moving_torso_thigh_angles', 'moving_thigh_shank_angles', 'shoulder_ankle_diffs']]
    X_bent_arm = feature_vector[['stationary_upper_arm_torso_angles', 'stationary_upper_arm_forearm_angles']]
    X_arm = feature_vector[['moving_upper_arm_torso_angles', 'moving_upper_arm_forearm_angles', 'shoulder_wrist_diffs']]

    X_leg_scaled = leg_scaler.transform(X_leg.values)
    X_bent_arm_scaled = bent_arm_scaler.transform(X_bent_arm.values)
    X_arm_scaled = arm_scaler.transform(X_arm.values)

    leg_preds = leg_model.predict(X_leg_scaled)
    bent_arm_preds = bent_arm_model.predict(X_bent_arm_scaled)
    arm_preds = arm_model.predict(X_arm_scaled)

    print("leg preds: {}".format(leg_preds))
    print("held_frames_idxs: {}".format(held_frames_idxs))

    if np.any(leg_preds == 1):
      mistakes.append("Raised leg too high")
      # Take first frame where error is detected
      extended_leg_too_high_frame_idx = np.where(leg_preds == 1)[0][0]
      print("extended_leg_too_high_frame_idx: {}".format(extended_leg_too_high_frame_idx))
      mistake_frame_dict["Raised leg too high"] = movenet_imgs[held_frames_idxs[extended_leg_too_high_frame_idx]]

    if np.any(leg_preds == 2):
      mistakes.append("Raised leg too low")
      # Take first frame where error is detected
      extended_leg_too_low_frame_idx = np.where(leg_preds == 2)[0][0]
      print("extended_leg_too_low_frame_idx: {}".format(extended_leg_too_low_frame_idx))
      mistake_frame_dict["Raised leg too low"] = movenet_imgs[held_frames_idxs[extended_leg_too_low_frame_idx]]

    if np.any(bent_arm_preds == 1):
      mistakes.append("Bent supporting arm")
      # Take first frame where error is detected
      bent_arm_frame_idx = np.where(bent_arm_preds == 1)[0][0]
      print("bent_arm_frame_idx: {}".format(bent_arm_frame_idx))
      mistake_frame_dict["Bent supporting arm"] = movenet_imgs[held_frames_idxs[bent_arm_frame_idx]]

    if np.any(arm_preds == 1):
      mistakes.append("Raised arm too high")
      # Take first frame where error is detected
      arm_frame_idx = np.where(arm_preds == 1)[0][0]
      mistake_frame_dict["Raised arm too high"] = movenet_imgs[held_frames_idxs[arm_frame_idx]]
  
    if np.any(arm_preds == 2):
      mistakes.append("Raised arm too low")
      # Take first frame where error is detected
      arm_frame_idx = np.where(arm_preds == 2)[0][0]
      mistake_frame_dict["Raised arm too low"] = movenet_imgs[held_frames_idxs[arm_frame_idx]]
  except:
    print("Error processing video")
    error_flag = True

  return mistakes, mistake_frame_dict, error_flag