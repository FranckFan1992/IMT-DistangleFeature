import torch
import torch.nn.functional as F


"""
6D rotation 是很多视觉模型使用的稳定表示（来自论文 On the Continuity of Rotation Representations in Neural Networks）。

"""

def compute_relative_pose(pose_i, pose_j):
    """
    pose_i: [B,4,4]
    pose_j: [B,4,4]

    return:
        R_ij [B,3,3]
        t_ij [B,3]
    """

    pose_i_inv = torch.inverse(pose_i)

    T_ij = torch.matmul(pose_j, pose_i_inv)

    R_ij = T_ij[:, :3, :3]
    t_ij = T_ij[:, :3, 3]

    # ⭐ translation normalization（非常重要）
    t_ij = t_ij / (torch.norm(t_ij, dim=1, keepdim=True) + 1e-8)

    return R_ij, t_ij


def rotation_matrix_to_6d(R):
    """
    R: [B,3,3]
    return: [B,6]
    """

    return R[:, :, :2].reshape(R.shape[0], 6)


def pose_matrix_to_9d(pose_i, pose_j):
    """
    pose_i: [B,4,4]
    pose_j: [B,4,4]

    return:
        pose_9d: [B,9]
    """

    R_ij, t_ij = compute_relative_pose(pose_i, pose_j)

    rot6d = rotation_matrix_to_6d(R_ij)

    pose_9d = torch.cat([rot6d, t_ij], dim=1)

    return pose_9d


