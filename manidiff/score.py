import torch
import torch
import torch.nn as nn
import os
from dataset import *
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
def l_nearest_neighbor_accuracy(set1, set2, l=1):
    """
    Calculate the l-nearest neighbor accuracy between two point sets.
    :param set1: Tensor of shape (N, 3) where N is the number of points in set 1.
    :param set2: Tensor of shape (M, 3) where M is the number of points in set 2.
    :param l: The number of nearest neighbors to consider.
    :return: l-NNA score.
    """
    set1 = set1[:, None, :]  # Shape (N, 1, 3)
    set2 = set2[None, :, :]  # Shape (1, M, 3)

    # Compute squared distances
    distances = torch.sum((set1 - set2) ** 2, dim=2)

    # Find the l nearest distances for each point in set1 to any point in set2
    min_l_distances_set1, _ = torch.topk(distances, k=l, largest=False, dim=1)
    min_l_distances_set1 = torch.mean(min_l_distances_set1, dim=1)

    # Find the l nearest distances for each point in set2 to any point in set1
    min_l_distances_set2, _ = torch.topk(distances, k=l, largest=False, dim=0)
    min_l_distances_set2 = torch.mean(min_l_distances_set2, dim=0)

    # Calculate the mean of these l nearest distances
    mean_dist_set1 = torch.mean(min_l_distances_set1)
    mean_dist_set2 = torch.mean(min_l_distances_set2)

    # Modified Chamfer Distance is the average of mean distances from set1 to set2 and set2 to set1
    score = (mean_dist_set1 + mean_dist_set2) / 2.0

    return score.item()
def calculate_emd_nnd(point_cloud1, point_cloud2, k=1):
    
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)

    
    _, indices1 = tree1.query(point_cloud2, k=k)
    _, indices2 = tree2.query(point_cloud1, k=k)

    
    emd_scores = []
    for idx, point in enumerate(point_cloud1):
        nearest_point_in_2 = point_cloud2[indices2[idx]]
        emd_score = wasserstein_distance([point][0], [nearest_point_in_2][0])
        emd_scores.append(emd_score)

    for idx, point in enumerate(point_cloud2):
        nearest_point_in_1 = point_cloud1[indices1[idx]]
        emd_score = wasserstein_distance([point][0], [nearest_point_in_1][0])
        emd_scores.append(emd_score)

    
    average_emd = sum(emd_scores) / len(emd_scores)
    return average_emd
def load_ply_as_tensor(filepath):
    pcd = o3d.io.read_point_cloud(filepath)  
    points = np.asarray(pcd.points)  
    tensor = torch.tensor(points, dtype=torch.float32)  
    return tensor
if __name__ == '__main__':
    def cd_l_nna(train_loader, ply_filename, n, l):
        """
        Evaluate the l-NNA average score over n batches from a train loader.

        :param train_loader: DataLoader that provides batches of training data.
        :param ply_filename: Path to the .ply file that contains the predicted point cloud.
        :param n: Number of batches to process.
        :param l: The number of nearest neighbors to consider in the accuracy calculation.
        :return: Average l-NNA score over n batches.
        """
        l_nna_scores = []

        # Load the predicted point cloud only once
        predicted_point_cloud = load_ply_as_tensor(ply_filename)
        predicted_point_cloud = predicted_point_cloud.float()  # Ensure type is float

        for i, batch in enumerate(train_loader):
            if i >= n:  # Process only the first n batches
                break

            # Extract and prepare the original data tensor
            original_data_tensor = batch['train_points'][0]
            original_data_tensor = original_data_tensor.float()  # Ensure type is float

            # Calculate the l-NNA score
            l_nna_score = l_nearest_neighbor_accuracy(predicted_point_cloud, original_data_tensor, l=l)
            l_nna_scores.append(l_nna_score)

            # print(f"Batch {i + 1} l-NNA Score: {l_nna_score}")

        # Calculate the average l-NNA score
        average_l_nna = sum(l_nna_scores) / len(l_nna_scores)
        return average_l_nna, sum(l_nna_scores)


    def emd_l_nna(train_loader, ply_filename, n, l):
        """
        Evaluate the l-NNA average score over n batches from a train loader.

        :param train_loader: DataLoader that provides batches of training data.
        :param ply_filename: Path to the .ply file that contains the predicted point cloud.
        :param n: Number of batches to process.
        :param l: The number of nearest neighbors to consider in the accuracy calculation.
        :return: Average l-NNA score over n batches.
        """
        l_nna_scores = []

        # Load the predicted point cloud only once
        predicted_point_cloud = load_ply_as_tensor(ply_filename)
        predicted_point_cloud = predicted_point_cloud.float()  # Ensure type is float

        for i, batch in enumerate(train_loader):
            if i >= n:  # Process only the first n batches
                break

            # Extract and prepare the original data tensor
            original_data_tensor = batch['train_points'][0]
            original_data_tensor = original_data_tensor.float()  # Ensure type is float

            # Calculate the l-NNA score
            l_nna_score = calculate_emd_nnd(predicted_point_cloud, original_data_tensor, k = 1)
            l_nna_scores.append(l_nna_score)

            # print(f"Batch {i + 1} l-NNA Score: {l_nna_score}")

        # Calculate the average l-NNA score
        average_l_nna = sum(l_nna_scores) / len(l_nna_scores)
        return average_l_nna, sum(l_nna_scores)

    # print(len(train_loader))
    a, b = cd_l_nna(train_loader, "airplane1.ply", len(train_loader)/10, 10)
    print('CD:', b)
    c, d = emd_l_nna(train_loader, "airplane1.ply", len(train_loader)/4, 10)
    print('EMD', d)
