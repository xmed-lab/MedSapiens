import argparse
import json
import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils.landmark_statistics import LandmarkStatistics

base_path = "../"

# Evaluation function
def evaluate(annotations_path, predictions_path,output_dir, keypoint_order=None, save_vis=False):
    with open(annotations_path, 'r') as f:
        gt_data = json.load(f)

    with open(predictions_path, 'rb') as f:
        pred_data = pickle.load(f)

    # Map ground truth annotations by image ID
    gt_annotations = {ann['image_id']: np.array(ann['keypoints']).reshape(-1, 3)[:, :2] for ann in
                      gt_data['annotations']}

    pred_annotations = {entry['img_path'].split('/')[-1]: np.array(entry['pred_instances']['keypoints']) for entry
                        in
                        pred_data}
    print(len(pred_annotations))

    landmark_statistic = LandmarkStatistics()
    for gt_image in tqdm(gt_data['images'], desc="Evaluating"):
        image_id = gt_image['id']
        file_name = gt_image['file_name']

        gt_keypoints = gt_annotations[image_id]
        if keypoint_order:
            pred_keypoints = pred_annotations[file_name][0][keypoint_order]
        else:
            pred_keypoints = pred_annotations[file_name][0]

        if 'hand' in annotations_path:
            radii = [2.0, 4.0, 10.0]
            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             normalization_factor=50, normalization_indizes=[1, 5])
        elif 'head' in annotations_path:
            radii = [2.0, 3.0, 4.0]
            spacing = [0.1, 0.1]  # Spacing in mm per pixel
            # print('Evaluating Head')
            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             spacing=spacing)
        elif 'chest' in annotations_path:
            radii = [3.0, 6.0, 9.0]

            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             spacing=[512 / gt_image['width'], 512 / gt_image['height']])
            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             spacing=[512 / gt_image['width'], 512 / gt_image['height']])

        elif 'legs' in annotations_path:
            radii = [2.0, 4.0, 10.0]
            meta_json = os.path.join(base_path, 'data/leg/meta_json', str(image_id) + '_meta.json')
            with open(meta_json, 'r') as file:
                data = json.load(file)
                # Extract the spacing if present
                spacing_tuple = list(data['spacing'])[::-1]  # Convert list to tuple for immutability

            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             spacing=spacing_tuple)
        elif 'surgery' in annotations_path:
            radii = [3.0, 6.0, 9.0]

            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             spacing=[512 / gt_image['width'], 512 / gt_image['height']])
            landmark_statistic.add_landmarks(image_id=image_id, predicted=pred_keypoints, groundtruth=gt_keypoints,
                                             spacing=[512 / gt_image['width'], 512 / gt_image['height']])
        else:
            raise NotImplementedError

        if save_vis:
            image_path = f"Images/{gt_image['file_name']}"  # Replace with the correct image path
            image = cv2.imread(os.path.join(base_path,'med_sapien', image_path))

            for x, y in gt_keypoints:
                cv2.circle(image, (int(x), int(y)), radius=15, color=(0, 255, 0), thickness=-1)

            for x, y in pred_keypoints:
                cv2.circle(image, (int(x), int(y)), radius=15, color=(255, 0, 0), thickness=-1)

            # Save the visualized image
            output_path = os.path.join(output_dir, f"{image_id}_visualized.jpg")
            plt.imsave(output_path, image)

    overview_string = landmark_statistic.get_overview_string(radii)
    return overview_string


def parse_results(raw_output):
    """
    Parse the raw output string to extract meaningful metrics, including percentages of outliers.

    Args:
        raw_output (str): The raw evaluation output string.

    Returns:
        dict: A dictionary of parsed metrics.
    """
    results = {}
    lines = raw_output.splitlines()

    # Initialize total number of GT points
    total_gt_points = 0

    # Extract total number of GT points
    for line in lines:
        if "valid gt" in line:
            total_gt_points = int(line.split("out of")[1].split("valid gt")[0].strip())
            break

    # Parse PE values and percentages
    for line in lines:
        if line.startswith("mean:"):
            results["MRE"] = float(line.split(":")[1].strip())
        elif line.startswith("std:"):
            results["std"] = float(line.split(":")[1].strip())
        # elif line.startswith("median:"):
        #     results["median"] = float(line.split(":")[1].strip())
        elif line.startswith("#outliers >= 2.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_2.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 3.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_3.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 4.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_4.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 6.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_6.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 9.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_9.0"] = 100 - (outlier_count / total_gt_points) * 100
        elif line.startswith("#outliers >= 10.0:"):
            outlier_count = int(line.split(":")[1].split("(")[0].strip())
            results["SDR_10.0"] = 100 - (outlier_count / total_gt_points) * 100

    return results


# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions against ground truth annotations.")
    parser.add_argument('--annotations', type=str, required=True,
                        help="Path to the ground truth annotations JSON file.")
    parser.add_argument('--predictions', type=str, required=True, help="Path to the model predictions pickle file.")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Directory to save evaluation metrics and visualizations.")
    parser.add_argument('--save-vis', action='store_true', help="Flag to save visualizations of predictions.")

    args = parser.parse_args()

    # prefixes = ["chest", "head", "hand", "legs"]
    # model_types = ["lora_sapiens", "sapiens", "med_sapien_lora_sapiens", "med_sapien_sapiens"]
    # if save_vis:
    #     os.makedirs(os.path.join(args.output_dir), exist_ok=True)

    raw_output = evaluate(
        annotations_path=args.annotations,
        predictions_path=args.predictions,
        output_dir=args.output_dir,
        keypoint_order=None,
        save_vis=args.save_vis
    )
    # Parse the raw output
    parsed_results = parse_results(raw_output)
    for key, value in parsed_results.items():
        print(f"{key}: {value}")
