import numpy as np
import math
from scipy.stats import zscore
from scipy.interpolate import UnivariateSpline
from scipy.signal import normalize

class DTW_Segmentation:
    def __init__(self, hparams):
        self.hparams = hparams
        self.wait_repetition_end = False

    def perform_dtw(self, x, y, scale_threshold):
        """
        ALGORITHM 1:
        Find latest subsequence of x that matches y.
        :param x: candidate sequence (C)
        :param y: query sequence (Q)
        :param scale_threshold: minimum scale the un-normalised subsequence needs to be to be considered for alignment
        :return: best_cost (subsequence (S) with lowest cost), Cbest (cost matrix of 5 & Q), Dbest (accumulative cost matrix of S & Q), best_path (DTW warp path between S & Q)
        """
        best_path = None
        best_cost = np.inf
        Cbest = None
        Dbest = None
        for i in reversed(range(0, len(x) - 2)):  # Start from latest data in x
            # if size of un-normalised subsequence is below threshold then ignore (z-norm takes scale out of the equation)
            if np.ptp(x[i:]) < scale_threshold:
                continue
            else:
                # Znorm this subsequence and get cost matrices
                C, D = self._get_cost_matrices(zscore(x[i:]), y)
                path = self._get_path(i, D)  # Get path of subsequence
                
                # Calculate cost, multiple x elements aligned to a single y element are summed and averaged
                pat_path = path[0] - i
                gt_path = path[1]
                total_cost = 0
                costs_arr = np.array([])

                for j in range(len(gt_path)):
                    costs_arr = np.append(costs_arr,
                                        C[gt_path[j], pat_path[j]])  # Note C[path[1],path [0]] is correct order
                    if j == len(gt_path) - 1 or gt_path[j] != gt_path[j + 1]:
                        total_cost += costs_arr.sum() / len(costs_arr)  # Average costs
                        # Give up if total cost is greater than cost so far
                        if total_cost / len(y) > best_cost:
                            total_cost = np.inf
                            break
                        costs_arr = np.array([])
                total_cost = total_cost / len(y)

                if total_cost < best_cost:
                    best_path = path
                    best_cost = total_cost
                    Cbest = C
                    Dbest = D
                
        return best_cost, Cbest, Dbest, best_path

    def distance_measure (self, x, y, pathx, pathy):
    # Perform distance measure between two sequences.
        total_cost = 0
        distances = np.array([])
        for i in range (len (pathy)):
            distances = np.append(distances, self._dist(x[pathx[i]], y[pathy[i]]))
            if i == len(pathy) - 1 or pathy[i] != pathy[i+1]:
                total_cost += distances.sum() / len (distances) # Average costs
                distances = np.array([])
        return total_cost / len(y)

    def _dist (self, x, y): # Manhattan/absolute distance function for DTW
        return abs(x - y)
    
    def _get_cost_matrices(self, x, y): # Get cost matrix (C) and accumulated cost matrix (D)
        r, c = len(x), len(y)
        r1, c1 = r - 1, c - 1
        D = np.zeros((c, r), dtype=np.float32)
        for i in range (c):
            for j in range(r):
                D[i, j] = self._dist(x[j], y[i])
        C = D.copy()
        for i in reversed(range(c)):
            for j in reversed(range(r)):
                if i == c1 and j == r1:
                    continue
                i1, j1 = i + 1, j + 1
                D[i, j] += min(
                    D[i1, j1] if i1 < c and j1 < r else np.inf, 
                    D[i, j1] if j1 < r else np.inf,
                    D[i, j] if i1 < c else np.inf)
        return (C, D)
    
    def _get_path(self, oi, D): # Get DTW path
        ilen, jlen = np.array(D.shape) - 1
        i, j = ilen, jlen
        p, q = [i], [j]
        while i > 0 or j > 0:
            i1, j1 = i - 1, j - 1
            tb = np.argmin(
                (D[i1, j1] if i1 >= 0 and j1 >= 0 else np.inf, 
                 D[i1, j] if i1 >= 0 and j >= 0 else np.inf,
                 D[i, j1] if i >= 0 and j1 >= 0 else np.inf))
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                i -= 1
            else:
                j -= 1
            p.insert(0, 1)
            q.insert(0, j)
        return np.array(q) + oi, np.array(p)
    
    def segment(self, is_last_point=False):
        """
        ALGORITHM 2: 
        Check for latest segmentation of the candidate sequence against the query sequence, if one exists.
        :param is_last_point: This is only set to True in segment offline () and should be left as False in other cases.
        """

        self.candidate = self._get_segment_candidates() # Get segment candidates to form the candidate sequence, a z-normed subsequence of this will be matched against self.query
        self.candidate_inds = self.candidate_inds + np.max([self.last_rep_ind, self.window_size_frame])
        
        if len(self.candidate) < 2:
            return False, # Not enough points to compare
    
        total_cost, costs, acc, warp_path = self.perform_dtw(self.candidate, self.query, self.min_candidate_scale)

        if self.wait_repetition_end or total_cost <= self.hparams["dtw_cost_threshold"]:
            if not self.wait_repetition_end:
                for i in range (len (self.query_nondtw_features)): # Loop features to check if distance is below threshold, reject segmentation if above
                    dist = self.distance_measure(self.candidate_features_unnorm[self.candidate_inds[warp_path[0]], i], self.query_nondtw_features[i], warp_path[0] - warp_path[0][0], warp_path[1])
                    if dist > self.dist_thresholds [i]:
                        return False, # Reject segmentation
                self.wait_repetition_end = True # We have now confirmed that this is a repetition of the Query exercise so now lets wait for future data to see if the DTW cost falls further (reduces early segmentation)

            if total_cost != np.inf:
                self.rep_end_costs = np.append(self.rep_end_costs, total_cost)
                self.rep_end_data.append([self.candidate_inds[warp_path[0]], warp_path[1]])
            arg_min = np.argmin(self.rep_end_costs)

            if is_last_point or self.wait_repetition_end and total_cost == np.inf or len(self.rep_end_costs) - arg_min > self.hparams["num_costs"]: # End of repetition found. DTW costs have not found a new minimum for "num_costs".
                res = self.rep_end_data[arg_min]
                self.last_rep_ind = self.rep_end_data[arg_min][0][-1] - self.hparams ["num_costs"] # Leave enough frames in the buffer to ensure a cubic spline can be mapped to the data
                self.rep_end_costs = np.array([])
                self.rep_end_data = []
                self.wait_repetition_end = False
                return True, res
            return False,
        else:
            return False,   

    def _preprocess_query(self):
        self.query_dir_vecs = self._get_dir_vecs(self.query_joints_positions)
        self.query_dir_vecs = self._rotate_dir_vecs_to_LCS(self.query_dir_vecs[:, :-3], self.query_dir_vecs[:, -3:]) # rotate dir vecs to create Local Coord System, rotates HipLeft to HipRight
        self.query_features_unnorm = self.get_features_from_query()
        self.query_dtw_spline, query_spline_velocity = self._get_spline_and_derivative_of_sequence(self.query_features_unnorm[:, 0])
        self.min_candidate_scale = np.ptp(self.query_dtw_spline)*self.hparams["candidate_scale_threshold"] # Get un-normalised peak to peak (min & max difference) of query
        self.query = self._reduce_query(query_spline_velocity)
        self.query_nondtw_features = np.array([self.query_features_unnorm[self.query_inds, i] for i in range(0, len(self.query_features_unnorm[0]))])

    def _get_dir_vecs(self, joints_positions):
        """ Get direction vectors from joint positions, assumes X1, Y1, Z1, X2, Y2, Z2,... order """
        dir_vecs = np.array([], dtype=float).reshape(len(joints_positions), 0)
        for conn in self.joint_connections:
            start_joint = self.joint_names.index(conn[0])*3
            start_pos = joints_positions[:, start_joint:start_joint+3]
            for joint in conn[1]:
                end_joint = self.joint_names.index(joint)*3
                end_pos = joints_positions[:, end_joint:end_joint+3]
                dir_vecs = np.hstack([dir_vecs, normalize(end_pos-start_pos)])
        return dir_vecs

    def _rotate_dir_vecs_to_LCS(self, dir_vecs, hips_dir):
        """ Rotate dir vecs to Local Coordinate System, Hipleft to HipRight dir is rotated to X axis, other dir_vecs are rotated proportionally around Y (vertical axis) """
        for i in range(len(dir_vecs)):
            rot = math.atan2(1, 0) - math.atan2(hips_dir[i][2], hips_dir[i][0]) # Get rotation to X axis. Note [1] is Y (vertical) axis hence hips_dir[i][2], same as Kinect coordinate system v
            for j in range(0, len(dir_vecs[i]), 3):
             dir_vecs[i][j:j+3] = self._y_rotation(dir_vecs[1][j:j+3], rot)
        return dir_vecs

    def _y_rotation(self, vector, theta):
        """ Rotates 3-D vector around y (vertical) axis, in radians """
        rotated_position = [vector[0] * np.sin(theta) + vector[2] * np.cos(theta), vector[1], vector[2] * np.sin(theta) - vector[0] * np.cos(theta)]
        return rotated_position

    def _get_features_from_query(self): # Gets DTW feature and distance features. Ranks the joint connections by amount of movement in order to automatically choose features
        if self.num_of_features > sum([len(x[1])*3 for x in self.joint_connections]) - 3:
            raise Exception("Number of features must not be greater than joint connections array")
        feats_sum_change = np.array([])
        for ind in range(self.query_dir_vecs.shape[1]): # Rank the features by change over time
            feats_sum_change = np.append(feats_sum_change, sum(abs(np.gradient(self.query_dir_vecs[:, ind]))))
        self.features_inds = feats_sum_change.argsort()[::-1][:self.num_of features] # First clement is used for DTW alignment
        self.dist_thresholds = ((feats_sum_change[self.features_inds] / feats_sum_change[self.features_inds].max()) * self.hparams["dist_multiplier"]) + self.hparams["dist_base"]
        
        return self.query_dir_vecs[:, self.features_inds]

    def _reduce_query(self, spline_velocity): # Store only the important points in the query_dtw_spline e.g. zero velocity crossings and the start and end points
        self.query_inds = self._find_zvc (spline_velocity, 0) # Find ZVCs of query spline's velocity
        self.query_inds = np.sort(np.unique(np.append(self.query_inds, [0, len(spline_velocity)-1]))) # Add first and last points of spline
        self.query_inds = self._remove_similar_points (self.query_dtw_spline, self.query_inds) # Remove similar points
        
        return zscore(self.query_dtw_spline[self.query_inds]) # data reduction and z-normalisation of query
    
    def _remove_similar_points (self, sequence, inds):
        inds_to_keep = np.array([inds[0]])
        for i in range (len (inds)-1):
            if abs (sequence[inds_to_keep[-1]]-sequence [inds [i+1]]) > self.hparams["reduction_threshold"]:
                inds_to_keep [-1] = inds[i] # Keep right most point
                inds_to_keep = np.append(inds_to_keep, inds [i+1]) # Add new point
        return inds_to_keep
            
    def _get_segment_candidates (self):
        self.candidate_dtw_spline, candidate_spline_velocity = self._get_spline_and_derivative_of_sequence(self.candidate_features_unnorm[np.max([self.last_rep_ind, self.window_size_frame]):, 0])
        self.candidate_inds = self._find_zvc(candidate_spline_velocity, self.hparams["velocity_cross_threshold"]) # Find ZVC of candidate spline’s velocity (first derivative of spline)
        self.candidate_inds = np.sort(np.unique(np.concatenate((self.candidate_inds,
                                                                 np.array([0, len(self.candidate_dtw_spline)-1]), # Include first and last points of candidate spline.
                                                                 self._get_indices_queryval(self.query_features_unnorm[:, 0][self.query_inds[0]], self.candidate_dtw_spline)))))# Include points from th
        self.candidate_inds = self._remove_similar_points(self.candidate_dtw_spline, self.candidate_inds) # Remove similar points
        return self.candidate_dtw_spline[self.candidate_inds] # Note z-norm only occurs on each subsequence of candidate

    def _get_indices_queryval(self, val, seq): # Get indices of sequence that pass through val
        res = np.array([], dtype=np.int)
        for i in range(1, len(seq)):
            if val >= seq[i - 1] and val <= seq[i] or val >= seq[i] and val <= seq[i - 1]:
                if abs(val - seq[i]) <= abs(val - seq[i - 1]):
                    res = np.append(res, i)
            else:
                res = np.append(res, i - 1)
        return res

    def _get_spline_and_derivative_of_sequence(self, sequence):
        sequence_inds = range(len(sequence))
        spline_func = UnivariateSpline(sequence_inds, sequence, s=self.hparams["spline_smoothing"], k=3) # Fit cubic spline
        spline = spline_func(sequence_inds)
        sequence_velocity_func = spline_func.derivative(n=1) # Get first derivative / Change in velocity
        spline_velocity = sequence_velocity_func(sequence_inds) # Sample first derivative
        return spline, spline_velocity

    def _find_zvc(self, seq, t=0.002): # Find zero velocity crossings
        res = np.array([], dtype=np.int)
        for i in range(len(seq) - 1):
            if seq[i + 1] > t > seq[i] or seq[i + 1] < t < seq[i] or seq[i + 1] > -t > seq[i] or seq[i + 1] < -t < seq[i]:
                res = np.append(res, 1)
        return res
