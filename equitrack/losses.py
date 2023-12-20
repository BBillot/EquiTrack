import torch
import numpy as np
from pytorch3d.transforms import matrix_to_euler_angles


def xfm_loss(true, pred, loss_type='l2', weight_R=5.e3, weight_T=1.e3):
    """Supervised loss between the predicted and GT transforms in homogeneous representations.
    The translation loss is always L2, be rotation can also be geodesic. Different weights can be assigned to
    translation/rotation parts."""

    # rotation loss
    true_R = true[:, :3, :3]
    pred_R = pred[:, :3, :3]
    if loss_type == 'l2':
        err_R = (true_R - pred_R) ** 2
    elif loss_type == 'geodesic':
        err_R = geodesic_distance(true_R, pred_R) ** 2
    else:
        raise ValueError('loss should either be l2 or geodesic, had %s' % loss_type)
    err_R = err_R.mean()

    # L2 loss for translation
    true_T = true[:, :3, 3]
    pred_T = pred[:, :3, 3]
    err_T = (true_T - pred_T) ** 2
    err_T = err_T.mean()

    return weight_R * err_R + weight_T * err_T


def image_loss(true, pred, loss_type='l2'):
    """Image voxel-wise loss. Can either be l2 or l1."""
    loss = true - pred
    if loss_type == 'l2':
        loss = torch.square(loss)
    elif loss_type == 'l1':
        loss = torch.abs(loss)
    else:
        ValueError('loss_type should be l2 (default) or l1, had %s' % loss_type)
    loss = torch.sum(loss, dim=[1, 2, 3, 4]).mean()
    return loss


def geodesic_distance(m1, m2):
    """Compute geodesic distance between two 3x3 rotation matrices in torch
    geo = arcos(0.5*(trace(M1@M2 - 1))), where @ is the matrix multiplication."""
    m = torch.matmul(m1, m2.transpose(1, 2))
    cos = (torch.einsum('bii->b', m) - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones_like(cos).to(device=cos.device)))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones_like(cos).to(device=cos.device)) * -1)
    cos = torch.clamp(cos, -1, 1)
    theta = torch.acos(cos)
    return theta


def dice(true, approx, eps=1e-7):
    """Dice metric between two binary masks (i.e., with 0/1 integers) of size [B, 1, H, W, D] in torch."""
    intersection = torch.sum(approx * true, [1, 2, 3, 4])
    cardinality = torch.sum(approx + true, [1, 2, 3, 4])
    return 2. * intersection / (cardinality + eps)


def rotation_matrix_to_angle_loss(true, pred):
    """L1 loss for rotations (in degrees) (useful to use as metric for testing)"""
    err_R = matrix_to_euler_angles(true, 'XYZ') - matrix_to_euler_angles(pred, 'XYZ')
    return torch.abs(err_R).mean() * 180 / np.pi


def translation_loss(true, pred):
    """L1 loss for translation (useful to use as metric for testing)"""
    return torch.abs(true - pred).mean()
