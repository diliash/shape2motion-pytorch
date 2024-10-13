import argparse
import json
import os

import h5py
import numpy as np
from tqdm import tqdm

CONFIDENCE_THRESHOLD = 0.8

def infer_semantic_label(normals, motion_dict):
    # Simple heuristic to infer semantic label, Z UP
    # If motion is translation - drawer
    # If motion is rotation and axis is vertical - door
    # If motion is rotation and axis is horizontal and normals point upwards on average - lid
    # If motion is rotation and axis is horizontal and normals point not upwards on average - door
    # revolute - 0, prismatic - 1
    # drawer - 0, door - 1, lid - 2, base - 3

    if motion_dict["mtype"] == 1:
        return 0
    if motion_dict["mtype"] == 0:
        axis_major_dir = np.argmax(np.abs(motion_dict["maxis"]))
        if axis_major_dir == 2:
            return 1
        else:
            avg_normal = np.mean(normals, axis=0)
            avg_normal_major_dir = np.argmax(np.abs(avg_normal))
            if avg_normal_major_dir == 2:
                return 2
            else:
                return 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="val_nms_result.h5")
    parser.add_argument("--output_path", type=str, default="shape2motion_pred_hssd_extended_subset_subset")
    parser.add_argument("--preprocessed_data_path", type=str, default="results/preprocess/val.h5")
    parser.add_argument("--original_subset_path", type=str)
    args = parser.parse_args()

    preprocess_h5_file = h5py.File(args.preprocessed_data_path, "r")
    original_subset_h5_file = h5py.File(args.original_subset_path, "r")
    model_id_idx_map = {}
    for model_id in original_subset_h5_file["model_ids"]:
        model_id_idx_map[model_id.decode()] = len(model_id_idx_map)

    h5_file = h5py.File(args.pred_path, "r")
    base_path = args.output_path
    os.makedirs(os.path.join(base_path, "predicted_masks"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "motion"), exist_ok=True)
    for model_id in tqdm(h5_file.keys()):
        correct_id = model_id.split("_")[0]
        text_path = os.path.join(base_path, f"{correct_id}.txt")
        instance_seg = h5_file[model_id]["pred_part_proposal"].astype(np.int32)
        final_map = -np.ones(np.shape(instance_seg))

        point_idx = preprocess_h5_file[model_id]["point_idx"]
        original_subset_normals = np.asarray(original_subset_h5_file["normals"][model_id_idx_map[correct_id]]).reshape((-1, 3))[point_idx]
        original_subset_points = np.asarray(original_subset_h5_file["points"][model_id_idx_map[correct_id]]).reshape((-1, 3))[point_idx]
        scores = h5_file[model_id]["pred_part_proposal_scores"]
        unique_ids = np.unique(instance_seg)
        txt_str = ""
        base_id = None
        base_mask = -np.ones(np.shape(instance_seg))
        for unique_id in unique_ids:
            motion_path = os.path.join(base_path, "motion", f"{correct_id}-{unique_id}.json")
            score = scores[unique_id]
            motion_idx = np.where(np.asarray(h5_file[model_id]["pred_joints_map"]) == unique_id)[0]
            if len(motion_idx) != 0 and score > CONFIDENCE_THRESHOLD:
                if len(unique_ids) > 1:
                    if float(score) > 0.001:
                        rel_path = os.path.join("predicted_masks", f"{correct_id}_" + '%03d' % unique_id + ".txt")
                        instance_idx = instance_seg == unique_id
                        final_map[instance_idx] = unique_id
                        np.savetxt(os.path.join(base_path, rel_path), instance_idx, fmt="%d")
                        
                        motion = np.asarray(h5_file[model_id]["pred_joints"][motion_idx])[0]
                        joint_origin = motion[:3]
                        joint_direction = motion[3:6]
                        joint_direction = joint_direction / np.linalg.norm(joint_direction)
                        joint_type = motion[6] - 1
                        motion_dict = {"mtype": joint_type.tolist(), "morigin": joint_origin.tolist(), "maxis": joint_direction.tolist()}
                        semantic_label = infer_semantic_label(original_subset_normals[instance_idx], motion_dict)
                        txt_str = txt_str + rel_path + " " + str(semantic_label) + " " + str(score) + "\n"
                        with open(motion_path, "w+") as json_file:
                            json.dump(motion_dict, json_file)
                else:
                    rel_path = os.path.join("predicted_masks", f"{correct_id}_" + '%03d' % unique_id + ".txt")
                    instance_idx = instance_seg == unique_id
                    final_map[instance_idx] = unique_id
                    txt_str = txt_str + rel_path + " " + str(0) + " " + str(score) + "\n"
                    np.savetxt(os.path.join(base_path, rel_path), instance_idx, fmt="%d")
                    motion = np.asarray(h5_file[model_id]["pred_joints"][motion_idx])[0]
                    joint_origin = motion[:3]
                    joint_direction = motion[3:6]
                    joint_direction = joint_direction / np.linalg.norm(joint_direction)
                    joint_type = motion[6] - 1
                    motion_dict = {"mtype": joint_type.tolist(), "morigin": joint_origin.tolist(), "maxis": joint_direction.tolist()}


        if (final_map == -1).any():
            base_id = len(unique_ids)
            while base_id in unique_ids:
                base_id += 1
            instance_idx = final_map == -1
            rel_path = os.path.join("predicted_masks", f"{correct_id}_" + '%03d' % base_id + ".txt")
            txt_str = txt_str + rel_path + " " + str(3) + " " + str(1) + "\n"
            np.savetxt(os.path.join(base_path, rel_path), instance_idx, fmt="%d")

        with open(text_path, "w") as f:
            f.write(txt_str)
    h5_file.close()
    h5_file.close()
