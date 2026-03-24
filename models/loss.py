from pathlib import Path
import configparser

import torch
import torch.nn.functional as F

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"
_CONFIG = configparser.ConfigParser()
_CONFIG.read(_CONFIG_PATH)
SAT_BEV_RES = _CONFIG.getint("Model", "sat_bev_res")

def _select_first_unique_pairs(primary_indices, secondary_indices, device):
    _, inverse, counts = torch.unique(primary_indices, sorted=True, return_inverse=True, return_counts=True)
    _, order = torch.sort(inverse, stable=True)
    first_positions = counts.cumsum(0)
    first_positions = torch.cat((torch.zeros(1, dtype=first_positions.dtype, device=device), first_positions[:-1]))
    keep = order[first_positions]
    return primary_indices[keep], secondary_indices[keep]

def scale_loss_log_l1(s_pred, s_gt, eps=1e-8):
    s_pred_clamped = s_pred.clamp(min=eps)
    s_gt_clamped = s_gt.clamp(min=eps)
    loss = torch.abs(torch.log(s_pred_clamped) - torch.log(s_gt_clamped))
    return loss.mean()

def entropy_loss(matching_score, eps=1e-8):

    row_entropy = - (matching_score * (matching_score + eps).log()).sum(dim=-1).mean()

    col_entropy = - (matching_score * (matching_score + eps).log()).sum(dim=-2).mean()

    return row_entropy + col_entropy

def mutual_nn_loss(matching_score):

    row_max, _ = matching_score.max(dim=-1)

    col_max, _ = matching_score.max(dim=-2)

    return -0.5 * (row_max.mean() + col_max.mean())

def loss_bev_space(X0, Rgt, tgt, R, t):
    B = X0.shape[0]

    X1_gt = Rgt @ X0.repeat(B,1,1).transpose(2, 1) + tgt.transpose(2, 1)
    X1_pred = R @ X0.repeat(B,1,1).transpose(2, 1) + t.transpose(2, 1)

    loss = torch.mean(torch.sqrt(((X1_gt - X1_pred)**2).sum(dim=1)), dim=-1)

    return loss

def trans_l1_loss(t, tgt):
    return torch.abs(t - tgt).sum(dim=-1)

def trans_l2_loss(t, tgt):
    return ((t[:, :, :2] - tgt[:, :, :2]) ** 2).sum(dim=-1)

def translation_direction_loss(t_pred, t_gt, s_pred, eps=1e-8):

    s_pred_detached = s_pred.detach().squeeze(-1).squeeze(-1)
    t_gt_scaled = t_gt.squeeze(1) / (s_pred_detached.unsqueeze(-1) + eps)

    t_pred_norm = F.normalize(t_pred.squeeze(1), dim=-1)
    t_gt_norm = F.normalize(t_gt_scaled, dim=-1)

    loss = 1 - (t_pred_norm * t_gt_norm).sum(dim=-1).mean()
    return loss

def rot_angle_loss(R, Rgt):
    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = torch.clip((trace - 1) / 2, -0.99999, 0.99999)
    R_err = torch.acos(cosine)
    return R_err.mean()

def compute_pose_loss(R, t, Rgt, tgt, soft_clipping=True):
    loss_rot, rot_err = rot_angle_loss(R, Rgt)
    loss_trans = trans_l1_loss(t, tgt)

    if soft_clipping:
        loss = torch.tanh(loss_rot / 0.9) + torch.tanh(loss_trans / 0.9)
    else:
        loss = loss_rot + loss_trans

    return loss, loss_rot, loss_trans

def vcre_loss(R, t, Tgt, K0, H=720):
    B = R.shape[0]
    Rgt, tgt = Tgt[:, :3, :3], Tgt[:, :3, 3:].transpose(1, 2)

    eye_coords = torch.from_numpy(eye_coords_glob).to(R.device, dtype=torch.float32).unsqueeze(0)[:, :, :3]
    eye_coords = eye_coords.expand(B, -1, -1)

    uv_gt = project_2d(eye_coords, K0)

    eye_coord_tmp = (R @ eye_coords.transpose(2, 1)) + t.transpose(2, 1)
    eyes_residual = (Rgt.transpose(2, 1) @ eye_coord_tmp - Rgt.transpose(2, 1) @ tgt.transpose(2, 1)).transpose(2, 1)

    uv_pred = project_2d(eyes_residual, K0)

    uv_gt = torch.clip(uv_gt, 0, H)
    uv_pred = torch.clip(uv_pred, 0, H)

    repr_err = torch.mean(torch.norm(uv_gt - uv_pred, dim=-1), dim=-1, keepdim=True)

    return repr_err

def compute_vcre_loss(R, t, Rgt, tgt, K=None, soft_clipping=True):
    B = R.shape[0]
    Tgt = torch.zeros((B, 4, 4), device=R.device, dtype=torch.float32)
    Tgt[:, :3, :3] = Rgt
    Tgt[:, :3, 3:] = tgt.transpose(2, 1)

    loss = vcre_loss(R, t, Tgt, K)

    if soft_clipping:
        loss = torch.tanh(loss / 80)

    loss_rot, rot_err = rot_angle_loss(R, Rgt)
    loss_trans = trans_l1_loss(t, tgt)

    return loss, loss_rot, loss_trans

def compute_similarity_loss(Rgt, tgt, sat_desc, grd_desc, sat_points_selected, grd_points_selected, sat_indices_sampled, grd_indices_sampled, coord_sat, coord_grd):

    B, num_points, _ = sat_points_selected.shape
    device = sat_desc.device

    ones = torch.ones(B, num_points, 1, device=device)
    sat_points_h = torch.cat((sat_points_selected, ones), dim=-1)
    grd_points_h = torch.cat((grd_points_selected, ones), dim=-1)

    T_s2g = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_s2g[:, :2, :2] = Rgt
    T_s2g[:, :2, 2] = tgt[:, 0, :]

    grd_points_mapped = (T_s2g @ sat_points_h.transpose(2, 1)).permute(0, 2, 1)
    grd_points_mapped[..., :2] /= grd_points_mapped[..., 2:3]
    grd_points_mapped = grd_points_mapped[..., :2]

    distances = torch.cdist(grd_points_mapped, coord_grd)
    min_distances, grd_indices_mapped = distances.min(dim=-1)

    keep_index = min_distances <= (grid_size_h / grd_bev_res)

    similarity_loss_s2g = torch.zeros(B, device=device)
    for b in range(B):
        if keep_index[b].sum() > 0:
            sat_indices_kept = sat_indices_sampled[b, keep_index[b]]
            grd_indices_kept = grd_indices_mapped[b, keep_index[b]]

            desc_similarity = torch.matmul(sat_desc[b, :, sat_indices_kept].T, grd_desc[b, :, grd_indices_kept])
            similarity_loss_s2g[b] = keep_index[b].sum() - torch.diagonal(desc_similarity).sum()

    T_g2s = torch.linalg.inv(T_s2g)
    sat_points_mapped = (T_g2s @ grd_points_h.transpose(2, 1)).permute(0, 2, 1)
    sat_points_mapped[..., :2] /= sat_points_mapped[..., 2:3]
    sat_points_mapped = sat_points_mapped[..., :2]

    distances = torch.cdist(sat_points_mapped, coord_sat)
    min_distances, sat_indices_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / SAT_BEV_RES)

    similarity_loss_g2s = torch.zeros(B, device=device)
    for b in range(B):
        if keep_index[b].sum() > 0:
            sat_indices_kept = sat_indices_mapped[b, keep_index[b]]
            grd_indices_kept = grd_indices_sampled[b, keep_index[b]]

            desc_similarity = torch.matmul(sat_desc[b, :, sat_indices_kept].T, grd_desc[b, :, grd_indices_kept])
            similarity_loss_g2s[b] = keep_index[b].sum() - torch.diagonal(desc_similarity).sum()

    return similarity_loss_s2g

def compute_infonce_loss_match_all_with_scale_select_negatives(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    mask, grid_size_h
):

    B, num_sat, num_grd = matching_score_original.shape
    scale = scale.to(torch.float32)
    scale = scale.detach()

    matches_row = matching_score_original.flatten(1)

    grd_points_mapped = ((sat_points_selected - tgt) / scale) @ Rgt

    distances = torch.cdist(grd_points_mapped, grd_coord)
    min_distances, col_mapped = distances.min(dim=-1)

    grd_pointwise_distance = torch.cdist(grd_coord, grd_coord)

    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b_distance = min_distances[b] <= (grid_size_h / SAT_BEV_RES) / scale[b]
        keep_index_b_depth = mask[b][col_mapped[b]]

        keep_index_b = keep_index_b_distance.squeeze(0) * keep_index_b_depth

        negative_mask = grd_pointwise_distance[b] > (1 / scale[b])

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_row = sampled_row[b, valid_indices]
            valid_col_mapped = col_mapped[b, valid_indices]

            unique_row, unique_col_mapped = _select_first_unique_pairs(
                valid_row,
                valid_col_mapped,
                matching_score_original.device,
            )

            negative_mask_for_unique_row = (negative_mask[unique_col_mapped]).to(torch.float32)

            selected_matching_indices = unique_row*num_grd + unique_col_mapped

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]) * negative_mask_for_unique_row, dim=1) + positives

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    sat_points_mapped = scale * (grd_points_selected @ Rgt.transpose(1, 2)) + tgt

    distances = torch.cdist(sat_points_mapped, sat_coord)
    min_distances, row_mapped = distances.min(dim=-1)

    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b = min_distances[b] <= (grid_size_h / SAT_BEV_RES)

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_col = sampled_col[b, valid_indices]
            valid_row_mapped = row_mapped[b, valid_indices]

            unique_col, unique_row_mapped = _select_first_unique_pairs(
                valid_col,
                valid_row_mapped,
                matching_score_original.device,
            )

            selected_matching_indices = unique_row_mapped*num_grd + unique_col

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def compute_infonce_loss_match_all_with_scale_select_negatives_homography(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    grid_size_h
):

    B, num_sat, num_grd = matching_score_original.shape
    scale = scale.to(torch.float32)
    scale = scale.detach()

    matches_row = matching_score_original.flatten(1)

    grd_points_mapped = ((sat_points_selected - tgt) / scale) @ Rgt

    distances = torch.cdist(grd_points_mapped, grd_coord)
    min_distances, col_mapped = distances.min(dim=-1)

    grd_pointwise_distance = torch.cdist(grd_coord, grd_coord)

    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b = min_distances[b] <= (grid_size_h / SAT_BEV_RES)

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_row = sampled_row[b, valid_indices]
            valid_col_mapped = col_mapped[b, valid_indices]

            unique_row, unique_col_mapped = _select_first_unique_pairs(
                valid_row,
                valid_col_mapped,
                matching_score_original.device,
            )

            selected_matching_indices = unique_row*num_grd + unique_col_mapped

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]))

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    sat_points_mapped = scale * (grd_points_selected @ Rgt.transpose(1, 2)) + tgt

    distances = torch.cdist(sat_points_mapped, sat_coord)
    min_distances, row_mapped = distances.min(dim=-1)

    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b = min_distances[b] <= (grid_size_h / SAT_BEV_RES)

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_col = sampled_col[b, valid_indices]
            valid_row_mapped = row_mapped[b, valid_indices]

            unique_col, unique_row_mapped = _select_first_unique_pairs(
                valid_col,
                valid_row_mapped,
                matching_score_original.device,
            )

            selected_matching_indices = unique_row_mapped*num_grd + unique_col

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def compute_infonce_loss_direction_only(
    Rgt, tgt,
    sat_points_selected, grd_points_selected,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    angle_threshold_deg=10.0,
    eps=1e-6
):

    B, num_sat, num_grd = matching_score_original.shape
    matching_score_exp = torch.exp(matching_score_original)

    angle_threshold_rad = angle_threshold_deg * torch.pi / 180.0

    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)

    sat_vectors = (sat_points_selected - tgt) @ Rgt
    sat_vectors = sat_vectors / (sat_vectors.norm(dim=-1, keepdim=True) + eps)

    grd_vectors = F.normalize(grd_coord, dim=-1)
    grd_vectors = grd_vectors / (grd_vectors.norm(dim=-1, keepdim=True) + eps)

    cos_sim = sat_vectors @ grd_vectors.transpose(1, 2)
    angles = torch.acos(torch.clamp(cos_sim, -1 + eps, 1 - eps))

    for b in range(B):

        positive_mask = (angles[b] <= angle_threshold_rad)
        negative_mask = ~positive_mask

        if positive_mask.sum() > 0:
            sampled_row_b = sampled_row[b]

            pos_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)
            neg_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)

            pos_index_mask[sampled_row_b] = positive_mask
            neg_index_mask[sampled_row_b] = negative_mask

            pos_scores = (pos_index_mask * matching_score_exp[b])[torch.unique(sampled_row_b), :]
            neg_scores = (neg_index_mask * matching_score_exp[b])[torch.unique(sampled_row_b), :]

            positives, _ = torch.max(pos_scores, dim=1)
            negatives = torch.sum(neg_scores, dim=1)

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / (positives + negatives)))
        else:
            infoNCE_loss_s2g[b] = 0.0

    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    grd_vectors = grd_points_selected @ Rgt.transpose(1, 2)
    grd_vectors = grd_vectors / (grd_vectors.norm(dim=-1, keepdim=True) + eps)

    sat_vectors = sat_coord - tgt
    sat_vectors = sat_vectors / (sat_vectors.norm(dim=-1, keepdim=True) + eps)

    cos_sim = sat_vectors @ grd_vectors.transpose(1, 2)
    angles = torch.acos(torch.clamp(cos_sim, -1 + eps, 1 - eps))

    for b in range(B):
        positive_mask = (angles[b] <= angle_threshold_rad)
        negative_mask = ~positive_mask

        all_zero_col_mask = positive_mask.sum(dim=0) == 0

        if positive_mask.sum() > 0:
            sampled_col_b = sampled_col[b]

            pos_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)
            neg_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)

            pos_index_mask[:, sampled_col_b] = positive_mask
            neg_index_mask[:, sampled_col_b] = negative_mask

            pos_scores = (pos_index_mask * matching_score_exp[b])[:, torch.unique(sampled_col_b[~all_zero_col_mask])]
            neg_scores = (neg_index_mask * matching_score_exp[b])[:, torch.unique(sampled_col_b[~all_zero_col_mask])]

            positives, _ = torch.max(pos_scores, dim=0)
            negatives = torch.sum(neg_scores, dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / (positives + negatives)))
        else:
            infoNCE_loss_g2s[b] = 0.0

    return (infoNCE_loss_s2g.mean() + infoNCE_loss_g2s.mean()) / 2

def compute_infonce_loss(
    Rgt, tgt,
    sat_points_selected, grd_points_selected,
    sampled_row, sampled_col,
    sat_indices_topk, grd_indices_topk,
    sat_keypoint_coord, grd_keypoint_coord,
    matching_score_original
):
    B, P, _ = sat_points_selected.shape
    device = matching_score_original.device

    ones = torch.ones(B, P, 1, device=device)
    sat_points_h = torch.cat([sat_points_selected, ones], dim=-1)
    grd_points_h = torch.cat([grd_points_selected, ones], dim=-1)

    T_s2g = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_s2g[:, :2, :2] = Rgt
    T_s2g[:, :2, 2] = tgt[:, 0, :]

    grd_points_mapped = (T_s2g @ sat_points_h.transpose(2, 1)).permute(0, 2, 1)
    grd_points_mapped[..., :2] /= grd_points_mapped[..., 2:3]
    grd_points_mapped = grd_points_mapped[..., :2]

    distances = torch.cdist(grd_points_mapped, grd_keypoint_coord)
    min_distances, col_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / grd_bev_res)

    infoNCE_loss_s2g = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = sampled_row[b, keep_index[b]]
            col_kept = col_mapped[b, keep_index[b]]

            unique, idx, counts = torch.unique(row_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            unique_row = row_kept[ind_sorted[cum_sum]]
            coorespond_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = unique_row * num_keypoints + coorespond_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    T_g2s = torch.linalg.inv(T_s2g)
    sat_points_mapped = (T_g2s @ grd_points_h.transpose(2, 1)).permute(0, 2, 1)
    sat_points_mapped[..., :2] /= sat_points_mapped[..., 2:3]
    sat_points_mapped = sat_points_mapped[..., :2]

    distances = torch.cdist(sat_points_mapped, sat_keypoint_coord)
    min_distances, row_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / SAT_BEV_RES)

    infoNCE_loss_g2s = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = row_mapped[b, keep_index[b]]
            col_kept = sampled_col[b, keep_index[b]]

            unique, idx, counts = torch.unique(col_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            coorespond_row = row_kept[ind_sorted[cum_sum]]
            unique_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = coorespond_row * num_keypoints + unique_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def compute_infonce_loss_with_mask(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    sat_indices_topk, grd_indices_topk,
    sat_keypoint_coord, grd_keypoint_coord,
    matching_score_original,
    sampled_mask
):
    B, P, _ = sat_points_selected.shape
    device = matching_score_original.device

    grd_points_mapped = scale * sat_points_selected @ Rgt.transpose(1, 2) + tgt

    distances = torch.cdist(grd_points_mapped, grd_keypoint_coord)
    min_distances, col_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / SAT_BEV_RES)

    keep_index = keep_index * sampled_mask

    infoNCE_loss_s2g = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = sampled_row[b, keep_index[b]]
            col_kept = col_mapped[b, keep_index[b]]

            unique, idx, counts = torch.unique(row_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            unique_row = row_kept[ind_sorted[cum_sum]]
            coorespond_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = unique_row * num_keypoints + coorespond_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    sat_points_mapped = ((grd_points_selected - tgt) @ Rgt) / scale

    distances = torch.cdist(sat_points_mapped, sat_keypoint_coord)
    min_distances, row_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / SAT_BEV_RES)
    keep_index = keep_index * sampled_mask

    infoNCE_loss_g2s = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = row_mapped[b, keep_index[b]]
            col_kept = sampled_col[b, keep_index[b]]

            unique, idx, counts = torch.unique(col_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            coorespond_row = row_kept[ind_sorted[cum_sum]]
            unique_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = coorespond_row * num_keypoints + unique_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def topology_direction_loss(Rgt, tgt,
    sat_points_selected, grd_points_selected, w, eps=1e-6):

    B, N, _ = sat_points_selected.shape

    sat_points_selected = sat_points_selected - tgt
    grd_points_selected = grd_points_selected @ Rgt.transpose(1, 2)

    W1 = torch.abs(w).sum(1, keepdim=True)
    w_norm = (w / (W1 + eps)).unsqueeze(-1)

    sat_mean = (w_norm * sat_points_selected).sum(1, keepdim=True)
    grd_mean = (w_norm * grd_points_selected).sum(1, keepdim=True)
    sat_centered = sat_points_selected - sat_mean
    grd_centered = grd_points_selected - grd_mean

    sat_norm = sat_centered / (sat_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1) + eps)
    grd_norm = grd_centered / (grd_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1) + eps)

    D_sat = torch.cdist(sat_norm, sat_norm)
    D_grd = torch.cdist(grd_norm, grd_norm)

    w_pair = torch.bmm(w.squeeze(-1).unsqueeze(2), w.squeeze(-1).unsqueeze(1))
    topo_error = (D_sat - D_grd) ** 2

    weighted_topo_error = (w_pair * topo_error).sum(dim=[1,2]) / (w_pair.sum(dim=[1,2]) + eps)
    topology_loss = weighted_topo_error.mean()

    v_sat = sat_points_selected / (sat_points_selected.norm(dim=-1, keepdim=True) + eps)
    v_grd = grd_points_selected / (grd_points_selected.norm(dim=-1, keepdim=True) + eps)
    cos_sim = (v_sat * v_grd).sum(dim=-1)

    direction_error = 1 - cos_sim
    weighted_direction_error = (w * direction_error).sum(dim=1) / (w.sum(dim=1) + eps)
    direction_loss = weighted_direction_error.mean()

    return topology_loss, direction_loss

def topology_ratio_direction_loss(Rgt, tgt,
    sat_points_selected, grd_points_selected, w, eps=1e-6, num_triplets=1000):

    B, N, _ = sat_points_selected.shape

    sat_points_selected = sat_points_selected - tgt
    grd_points_selected = grd_points_selected @ Rgt.transpose(1, 2)

    w = w.squeeze(-1)
    w_probs = w / (w.sum(dim=1, keepdim=True) + eps)

    ratio_errors = []

    for b in range(B):
        x_sat = sat_points_selected[b]
        x_grd = grd_points_selected[b]
        w_b = w_probs[b]

        i_idx = torch.multinomial(w_b, num_triplets, replacement=True)
        j_idx = torch.multinomial(w_b, num_triplets, replacement=True)
        k_idx = torch.multinomial(w_b, num_triplets, replacement=True)

        mask_valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        if mask_valid.sum() == 0:
            continue
        i_idx = i_idx[mask_valid]
        j_idx = j_idx[mask_valid]
        k_idx = k_idx[mask_valid]

        xi_sat, xj_sat, xk_sat = x_sat[i_idx], x_sat[j_idx], x_sat[k_idx]
        xi_grd, xj_grd, xk_grd = x_grd[i_idx], x_grd[j_idx], x_grd[k_idx]

        d_ij_sat = (xi_sat - xj_sat).norm(dim=-1)
        d_ik_sat = (xi_sat - xk_sat).norm(dim=-1)
        d_ij_grd = (xi_grd - xj_grd).norm(dim=-1)
        d_ik_grd = (xi_grd - xk_grd).norm(dim=-1)

        ratio_sat = d_ij_sat / (d_ik_sat + eps)
        ratio_grd = d_ij_grd / (d_ik_grd + eps)

        ratio_error = (ratio_sat - ratio_grd) ** 2
        ratio_errors.append(ratio_error.mean())

    if len(ratio_errors) > 0:
        topology_loss = torch.stack(ratio_errors).mean()
    else:
        topology_loss = torch.tensor(0.0, device=sat_points_selected.device)

    v_sat = sat_points_selected / (sat_points_selected.norm(dim=-1, keepdim=True) + eps)
    v_grd = grd_points_selected / (grd_points_selected.norm(dim=-1, keepdim=True) + eps)
    cos_sim = (v_sat * v_grd).sum(dim=-1)

    direction_error = 1 - cos_sim
    weighted_direction_error = (w * direction_error).sum(dim=1) / (w.sum(dim=1) + eps)
    direction_loss = weighted_direction_error.mean()

    return topology_loss, direction_loss
