import os
import numpy as np
import pickle
from scipy.signal import medfilt
from datetime import datetime
def evaluate_pose(pose_seq, exercise, pkl_path):
    """Evaluate a pose sequence for a particular exercise.

    Args:
        pose_seq: PoseSequence object.
        exercise: String name of the exercise to evaluate.

    Returns:
        correct: Bool whether exercise was performed correctly.
        feedback: Feedback string.

    """
    if exercise == 'bird_dog':
        return _bird_dog(pose_seq, pkl_path)
    else:
        return (False, "Exercise string not recognized.")

def _bird_dog(pose_seq, pkl_path):
    # find the arm and leg that is currently used
    now = datetime.now().strftime('%H_%M_%f')

    poses = pose_seq.poses

    joints_right = [{
                        "neck" : pose.neck,
                        "shoulder" : pose.rshoulder, 
                        "elbow" : pose.relbow, 
                        "wrist" : pose.rwrist, 
                        "hip" : pose.rhip, 
                        "midhip" : pose.midhip, 
                        "knee" : pose.rknee, 
                        "ankle" : pose.rankle 
                        # "heel" : pose.rheel, 
                        # "big toe" : pose.rbigtoe, 
                        # "small toe" : pose.rsmalltoe
                    } for pose in poses]
    joints_left = [{
                        "neck" : pose.neck,
                        "shoulder" : pose.lshoulder, 
                        "elbow" : pose.lelbow, 
                        "wrist" : pose.lwrist, 
                        "hip" : pose.lhip, 
                        "midhip" : pose.midhip, 
                        "knee" : pose.lknee, 
                        "ankle" : pose.lankle
                        # "heel" : pose.lheel, 
                        # "big toe" : pose.lbigtoe, 
                        # "small toe" : pose.lsmalltoe
                    } for pose in poses]
    # joints_mid = [{
    #                 "nose" : pose.nose, 
    #                 "neck" : pose.neck, 
    #                 "hip" : pose.midhip
    #             } for pose in poses]

    try:
        os.mkdir('test_exist/')
    except:
        pass
    try:
        with open('test_exist/right_joints_{}.csv'.format(now), 'w') as file:
            file.write("neck, shoulder, elbow, wrist, hip, midhip, knee, ankle\n")
    except:
        pass
    with open('test_exist/right_joints_{}.csv'.format(now), 'a') as file:
        for joint in joints_right:
            for part in joint:
                file.write('{}, '.format(joint[part].exists))
            file.write('\n')
    try:
        with open('test_exist/left_joints_{}.csv'.format(now), 'w') as file:
            file.write("neck, shoulder, elbow, wrist, hip, midhip, knee, ankle\n")
    except:
        pass
    with open('test_exist/left_joints_{}.csv'.format(now), 'a') as file:
        for joint in joints_left:
            for part in joint:
                file.write('{}, '.format(joint[part].exists))
            file.write('\n')

    # filter out data points where a part does not exist
    joints_right = [joint for joint in joints_right if all(joint[part].exists for part in joint)]
    # joints_right_ = np.array(joints_right)
    joints_left = [joint for joint in joints_left if all(joint[part].exists for part in joint)]
    # joints_left_ = np.array(joints_left)
    # joints_mid = [joint for joint in joints_mid if all(joint[part].exists for part in joint)]
    # joints_mid_ = np.array(joints_mid)

    # elbow to shoulder
    right_upper_arm_vecs = np.array([(joint["shoulder"].x - joint["elbow"].x, joint["shoulder"].y - joint["elbow"].y) for joint in joints_right])
    # hip to shoulder
    right_torso_vecs = np.array([(joint["shoulder"].x - joint["hip"].x, joint["shoulder"].y - joint["hip"].y) for joint in joints_right])
    # midhip to neck
    right_mid_torso_vecs = np.array([(joint["neck"].x - joint["midhip"].x, joint["neck"].y - joint["midhip"].y) for joint in joints_right])
    # wrist to elbow
    right_forearm_vecs = np.array([(joint["elbow"].x - joint["wrist"].x, joint["elbow"].y - joint["wrist"].y) for joint in joints_right])
    # hip to knee
    right_thigh_vecs = np.array([(joint["knee"].x - joint["hip"].x, joint["knee"].y - joint["hip"].y) for joint in joints_right])
    # ankle to knee
    right_shank_vecs = np.array([(joint["knee"].x - joint["ankle"].x, joint["knee"].y - joint["ankle"].y) for joint in joints_right])
    
    # elbow to shoulder
    left_upper_arm_vecs = np.array([(joint["shoulder"].x - joint["elbow"].x, joint["shoulder"].y - joint["elbow"].y) for joint in joints_left])
    # hip to shoulder
    left_torso_vecs = np.array([(joint["shoulder"].x - joint["hip"].x, joint["shoulder"].y - joint["hip"].y) for joint in joints_left])
    # midhip to neck
    left_mid_torso_vecs = np.array([(joint["neck"].x - joint["midhip"].x, joint["neck"].y - joint["midhip"].y) for joint in joints_left])
    # wrist to elbow
    left_forearm_vecs = np.array([(joint["elbow"].x - joint["wrist"].x, joint["elbow"].y - joint["wrist"].y) for joint in joints_left])
    # hip to knee
    left_thigh_vecs = np.array([(joint["knee"].x - joint["hip"].x, joint["knee"].y - joint["hip"].y) for joint in joints_left])
    # ankle to knee
    left_shank_vecs = np.array([(joint["knee"].x - joint["ankle"].x, joint["knee"].y - joint["ankle"].y) for joint in joints_left])


    # mid_torso_vecs = np.array([(joint["neck"].x - joint["hip"].x, joint["neck"].y - joint["hip"].y) for joint in joints_mid])

    # normalize vectors
    right_upper_arm_vecs = right_upper_arm_vecs / np.expand_dims(np.linalg.norm(right_upper_arm_vecs, axis=1), axis=1)
    right_torso_vecs = right_torso_vecs / np.expand_dims(np.linalg.norm(right_torso_vecs, axis=1), axis=1)
    right_mid_torso_vecs = right_mid_torso_vecs / np.expand_dims(np.linalg.norm(right_mid_torso_vecs, axis=1), axis=1)
    right_forearm_vecs = right_forearm_vecs / np.expand_dims(np.linalg.norm(right_forearm_vecs, axis=1), axis=1)
    right_thigh_vecs = right_thigh_vecs / np.expand_dims(np.linalg.norm(right_thigh_vecs, axis=1), axis=1)
    right_shank_vecs = right_shank_vecs / np.expand_dims(np.linalg.norm(right_shank_vecs, axis=1), axis=1)

    left_upper_arm_vecs = left_upper_arm_vecs / np.expand_dims(np.linalg.norm(left_upper_arm_vecs, axis=1), axis=1)
    left_torso_vecs = left_torso_vecs / np.expand_dims(np.linalg.norm(left_torso_vecs, axis=1), axis=1)
    left_mid_torso_vecs = left_mid_torso_vecs / np.expand_dims(np.linalg.norm(left_mid_torso_vecs, axis=1), axis=1)
    left_forearm_vecs = left_forearm_vecs / np.expand_dims(np.linalg.norm(left_forearm_vecs, axis=1), axis=1)
    left_thigh_vecs = left_thigh_vecs / np.expand_dims(np.linalg.norm(left_thigh_vecs, axis=1), axis=1)
    left_shank_vecs = left_shank_vecs / np.expand_dims(np.linalg.norm(left_shank_vecs, axis=1), axis=1)

    # mid_torso_vecs = mid_torso_vecs / np.expand_dims(np.linalg.norm(mid_torso_vecs, axis=1), axis=1)

    # calculate angles between body parts
    right_upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_upper_arm_vecs, right_mid_torso_vecs), axis=1), -1.0, 1.0)))
    right_upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_upper_arm_vecs, -right_forearm_vecs), axis=1), -1.0, 1.0)))
    right_torso_thigh_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_mid_torso_vecs,  right_thigh_vecs), axis=1), -1.0, 1.0)))
    right_thigh_shank_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(right_thigh_vecs, right_shank_vecs), axis=1), -1.0, 1.0)))
    
    left_upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_upper_arm_vecs, left_mid_torso_vecs), axis=1), -1.0, 1.0)))
    left_upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_upper_arm_vecs, -left_forearm_vecs), axis=1), -1.0, 1.0)))
    left_torso_thigh_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_mid_torso_vecs, left_thigh_vecs), axis=1), -1.0, 1.0)))
    left_thigh_shank_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(left_thigh_vecs, left_shank_vecs), axis=1), -1.0, 1.0)))

    print_dict = {       
        "right_upper_arm_torso_angles" : right_upper_arm_torso_angles,
        "right_upper_arm_forearm_angle" : right_upper_arm_forearm_angles,
        "right_torso_thigh_angles" : right_torso_thigh_angles,
        "right_thigh_shank_angles" : right_thigh_shank_angles,        
        "left_upper_arm_torso_angles" : left_upper_arm_torso_angles,
        "left_upper_arm_forearm_angles" : left_upper_arm_forearm_angles,
        "left_torso_thigh_angles" : left_torso_thigh_angles,
        "left_thigh_shank_angles" : left_thigh_shank_angles        
    }

    filtered_dict = {}
    for key in print_dict.keys():
        filtered_dict[key] = medfilt(medfilt(print_dict[key], 5), 5)


    # print(print_dict)
    with open(f"{pkl_path}.pkl", 'wb') as f:
        pickle.dump(print_dict, f)
    with open(f"{pkl_path}_filtered.pkl", 'wb') as f:
        pickle.dump(filtered_dict, f)

    right_upper_arm_torso_angles_max = np.max(medfilt(medfilt(right_upper_arm_torso_angles, 5), 5))
    right_upper_arm_torso_angles_min = np.min(medfilt(medfilt(right_upper_arm_torso_angles, 5), 5))
    right_upper_arm_forearm_angles_max = np.max(medfilt(medfilt(right_upper_arm_forearm_angles, 5), 5))
    right_upper_arm_forearm_angles_min = np.min(medfilt(medfilt(right_upper_arm_forearm_angles, 5), 5))
    right_torso_thigh_angles_max = np.max(medfilt(medfilt(right_torso_thigh_angles, 5), 5))
    right_torso_thigh_angles_min = np.min(medfilt(medfilt(right_torso_thigh_angles, 5), 5))
    right_thigh_shank_angles_max = np.max(medfilt(medfilt(right_thigh_shank_angles, 5), 5))
    right_thigh_shank_angles_min = np.min(medfilt(medfilt(right_thigh_shank_angles, 5), 5))
    left_upper_arm_torso_angles_max = np.max(medfilt(medfilt(left_upper_arm_torso_angles, 5), 5))
    left_upper_arm_torso_angles_min = np.min(medfilt(medfilt(left_upper_arm_torso_angles, 5), 5))
    left_upper_arm_forearm_angles_max = np.max(medfilt(medfilt(left_upper_arm_forearm_angles, 5), 5))
    left_upper_arm_forearm_angles_min = np.min(medfilt(medfilt(left_upper_arm_forearm_angles, 5), 5))
    left_torso_thigh_angles_max = np.max(medfilt(medfilt(left_torso_thigh_angles, 5), 5))
    left_torso_thigh_angles_min = np.min(medfilt(medfilt(left_torso_thigh_angles, 5), 5))
    left_thigh_shank_angles_max = np.max(medfilt(medfilt(left_thigh_shank_angles, 5), 5))
    left_thigh_shank_angles_min = np.min(medfilt(medfilt(left_thigh_shank_angles, 5), 5))

    correct = True
    feedback = ''

    # print("right_upper_arm_torso_angles: min: {}, max: {}".format(right_upper_arm_torso_angles_min, right_upper_arm_torso_angles_max))
    # print("right_upper_arm_forearm_angles: min: {}, max: {}".format(right_upper_arm_forearm_angles_min, right_upper_arm_forearm_angles_max))
    # print("right_torso_thigh_angles: min: {}, max: {}".format(right_torso_thigh_angles_min, right_torso_thigh_angles_max))
    # print("right_thigh_shank_angles_min: min: {}, max: {}".format(right_thigh_shank_angles_min, right_thigh_shank_angles_max))
    # print("left_upper_arm_torso_angles: min: {}, max: {}".format(left_upper_arm_torso_angles_min, left_upper_arm_torso_angles_max))
    # print("left_upper_arm_forearm_angles: min: {}, max: {}".format(left_upper_arm_forearm_angles_min, left_upper_arm_forearm_angles_max))
    # print("left_torso_thigh_angles: min: {}, max: {}".format(left_torso_thigh_angles_min, left_torso_thigh_angles_max))
    # print("left_thigh_shank_angles_min: min: {}, max: {}".format(left_thigh_shank_angles_min, left_thigh_shank_angles_max))

    if correct:
        return (correct, 'Exercise performed correctly!')
    else:
        return (correct, feedback)
    