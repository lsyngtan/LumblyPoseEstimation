import os
import pickle

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline
from scipy.stats import zscore

from dtaidistance.subsequence.dtw import subsequence_alignment
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance.subsequence.dtw import subsequence_search
from dtaidistance import dtw_ndim, dtw

import librosa
import libfmp.b
import libfmp.c4
import libfmp.c7


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
  vector_dict, angle_dict, filtered_dict = _bird_dog(pose_seq)

  return vector_dict, angle_dict, filtered_dict

def extract_exemplar_and_query(exemplar_keypts, query_keypts):
  """
  Splits each video by rep count -> exemplar = first rep; queries = additional reps after this first rep
  """
  exemplar_vectors, _, _ = preprocess_pose(exemplar_keypts)
  query_vectors, _, filtered_query_dict = preprocess_pose(query_keypts)

  return exemplar_vectors, query_vectors, filtered_query_dict

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

def perform_subseq_dtw(exemplar_spline, query_spline, k = 10):
  """
  Extract matching "reps" using subsequence DTW
  """
  sa = subsequence_alignment(
     zscore(exemplar_spline), 
     zscore(query_spline)
  )
  matches = sa.kbest_matches(k)
  idxs = [m.segment for m in matches]
  
  return idxs

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

def perform_segmentation(exemplar_keypts, query_keypts):
  """
  Segments repetitions using DTW:
  exemplar_keypts: keypoints for exemplar video (sample rep taken during "calibration")
  query_keypts: - keypoints for video we want to analyze (10 reps)
  Ideally should be of the same person
  """
  # Extract exemplar as first rep, query as subsequent reps
  exemplar_vector, query_vectors, filtered_query_dict = extract_exemplar_and_query(exemplar_keypts, query_keypts)
  
  # Calculate total change associated w/ each motion feature and rank
  FRS = get_FRS(exemplar_vector)
  FRS = FRS.sort_values(by = 'FR', ascending = False)

  # Spline fitting
  ft_name, exemplar_feature = get_top_ranking_motion_ft(FRS, exemplar_vector)
  xs, exemplar_spline, dt = get_DTW_spline(exemplar_feature)
  
  ft_name, query_feature = get_top_ranking_motion_ft(FRS, query_vectors)
  xs, query_spline, dt = get_DTW_spline(query_feature)

  idxs = perform_subseq_dtw_alt(exemplar_spline, query_spline)

  # Plot algorithmic segmentations
  fig, ax = plt.subplots(figsize=(20, 8))
  ax.plot(zscore(query_spline), label = 'Candidate Sequence', linestyle = ':')

  for idx_set in idxs:
    ax.axvline(idx_set[0], linestyle = '--', color = 'blue')
    ax.axvline(idx_set[1], linestyle = '--', color = 'blue')
    ax.axvspan(idx_set[0], idx_set[1], alpha = 0.5, color = 'red')
  ax.plot(zscore(exemplar_spline), label = 'Exemplar Sequence', color = 'black')
  ax.set_ylabel(f"{ft_name} Unit Vector")
  ax.legend()
  ax.set_title("Rep Segmentations using Subsequence DTW Video")

  return idxs, filtered_query_dict

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

def construct_dataframe(idxs, filtered_dict, offset = 0):
  # Construct dataframe for correct reps
  rep_df = pd.DataFrame(filtered_dict)

  rep_dfs = []
  for n_reps, idx_set in enumerate(idxs):
    df = rep_df.iloc[idx_set[0]:idx_set[1], :]
    df['Rep'] = n_reps + offset
    rep_dfs.append(df)
  rep_df = pd.concat(rep_dfs)

  return rep_df

def get_rom(rep_df, limb_keys):
  roms_dict = {}
  roms = [
      rep_df.groupby('Rep').agg({f"right_{limb_keys}": np.ptp}).mean(),
      rep_df.groupby('Rep').agg({f"left_{limb_keys}": np.ptp}).mean()
  ]
  for rom in roms:
    roms_dict[rom.index.values[0]] =  rom.values[0]
  
  moving_limb = max(roms_dict, key=roms_dict.get)
  moving_limb = moving_limb.split('_')[0]
  
  return moving_limb

def detect_moving_limbs(rep_df, arm_keys = 'upper_arm_torso_angles', 
                        leg_keys = 'torso_thigh_angles'):
  moving_arm = get_rom(rep_df, limb_keys = arm_keys)
  moving_leg = get_rom(rep_df, limb_keys = leg_keys)

  moving_sides = {"moving_arm": moving_arm, "moving_leg": moving_leg}
  print(moving_sides)
  
  return moving_sides

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

RELEVANT_COLS = [
    'moving_upper_arm_torso_angles',
    'stationary_upper_arm_forearm_angles',
    'moving_torso_thigh_angles',
    'moving_upper_arm_forearm_angles',
    'stationary_torso_thigh_angles',
    'stationary_upper_arm_torso_angles',
    'Rep',
]

# Averaged across all reps in training data w/ that particular mistake
SUPPORTING_ARM_THRESHES = (3.266140, 11.288011)
LEG_TORSO_THRESH = 70.090967

def load_data(exemplar_keypts, query_keypts):
  """
  Takes in exemplar + query keypoints from a video and outputs a dataframe 
  containing angular features for all repetitions
  """
  data = []
  # Segment reps 
  idxs, filtered_dict = perform_segmentation(exemplar_keypts, query_keypts)

  # Convert to dataframe
  rep_df = construct_dataframe(idxs, filtered_dict)

  # Detect moving limbs
  moving_limbs_df = detect_moving_limbs(rep_df)

  # Rename based on moving/stationary limbs
  rename_dict = rename_angle_columns(rep_df, moving_limbs_df)
  rep_df = rep_df.rename(columns = rename_dict)
  
  # Get relevant angular features
  rep_df = rep_df[RELEVANT_COLS]
  full_df = pd.concat(data)

  return full_df

# Output from MoveNet goes here:
def main(exemplar_keypts, query_keypts):
  mistakes = []

  # Load in data from video
  sample_df = load_data(exemplar_keypts, query_keypts)

  # Group by rep
  sample_df_grouped = sample_df.groupby("Rep")

  # Check for bent supporting arm 
  supporting_arm_df = sample_df_grouped.agg({
   "stationary_upper_arm_forearm_angles": ['min', 'max']
  }).mean()
  supporting_min_angle = supporting_arm_df['stationary_upper_arm_forearm_angles']['min']
  # supporting_max_angle = supporting_arm_df['stationary_upper_arm_forearm_angles']['max']
  if supporting_min_angle >= SUPPORTING_ARM_THRESHES[1]:
    mistakes.append("Supporting arm bent")

  # Check for leg too high/low
  torso_thigh_df = sample_df_grouped.agg({
     "moving_torso_thigh_angles": np.ptp
  }).mean()
  thigh_rom = torso_thigh_df["moving_torso_thigh_angles"]["ptp"]
  if thigh_rom < LEG_TORSO_THRESH:
     mistakes.append("Leg too low")
     
  return mistakes

