import torch
import torch.nn.functional as F

def desc_l2norm(desc: torch.Tensor) -> torch.Tensor:
    return F.normalize(desc, p=2, dim=1, eps=1e-10)

def weighted_procrustes_2d(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):

    assert len(A) == len(B)

    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        A_mean, B_mean = (w_norm * A).sum(1, keepdim=True), (w_norm * B).sum(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean

        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean, B_mean = A.mean(1, keepdim=True), B.mean(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    if check_rank and (torch.linalg.matrix_rank(H) == 1).sum() > 0:
        return None, None, False

    U, S, V = torch.svd(H)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)
    t = B_mean - A_mean @ R.transpose(1, 2)

    return R, t, True

def weighted_procrustes_2d_with_scale(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):
    assert len(A) == len(B)

    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)

        A_mean = (w_norm * A).sum(1, keepdim=True)
        B_mean = (w_norm * B).sum(1, keepdim=True)
        A_c = A - A_mean
        B_c = B - B_mean

        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean = A.mean(1, keepdim=True)
        B_mean = B.mean(1, keepdim=True)
        A_c = A - A_mean
        B_c = B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    if check_rank and (torch.linalg.matrix_rank(H) == 1).sum() > 0:
        return None, None, None, False

    U, S, V = torch.svd(H)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)

    A_c_rot = A_c @ R.transpose(1, 2)
    if use_weights:
        numerator = (w_norm * B_c * A_c_rot).sum(dim=1, keepdim=True).sum(-1, keepdim=True)
        denominator = (w_norm * A_c_rot ** 2).sum(dim=1, keepdim=True).sum(-1, keepdim=True) + eps
    else:
        numerator = (B_c * A_c_rot).sum(dim=1, keepdim=True).sum(-1, keepdim=True)
        denominator = (A_c_rot ** 2).sum(dim=1, keepdim=True).sum(-1, keepdim=True) + eps

    s = numerator / denominator

    t = B_mean - s * (A_mean @ R.transpose(1, 2))

    return R, t, s, True

def soft_inlier_counting_bev(X0, X1, R, t, th=50):
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)

def soft_inlier_counting_bev_with_scale(X0, X1, R, t, scale, th=50):
    beta = 5 / th
    X0_to_1 = scale * (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)

def inlier_counting_bev(X0, X1, R, t, th=50):
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()

def inlier_counting_bev_with_scale(X0, X1, R, t, scale, th=50):
    X0_to_1 = scale * (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()

class e2eProbabilisticProcrustesSolver():
    def __init__(self, it_RANSAC, it_matches, num_samples_matches, num_corr_2d_2d, num_refinements, th_inlier, th_soft_inlier, metric_coord_sat_B, metric_coord_grd_B):

        self.it_RANSAC = it_RANSAC
        self.it_matches = it_matches
        self.num_samples_matches = num_samples_matches
        self.num_corr_2d_2d = num_corr_2d_2d
        self.num_refinements = num_refinements
        self.th_inlier = th_inlier
        self.th_soft_inlier = th_soft_inlier
        self.metric_coord_sat_B = metric_coord_sat_B
        self.metric_coord_grd_B = metric_coord_grd_B

    def estimate_pose(self, matching_score, return_inliers=False):
        device = matching_score.device
        matches = matching_score.detach()

        B, num_kpts_sat, num_kpts_grd = matches.shape

        matches_row = matches.reshape(B, num_kpts_sat*num_kpts_grd)
        batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, self.num_samples_matches]).reshape(B, self.num_samples_matches)
        batch_idx_ransac = torch.tile(torch.arange(B).view(B, 1), [1, self.num_corr_2d_2d]).reshape(B, self.num_corr_2d_2d)

        num_valid_h = 0
        Rs = torch.zeros((B, 0, 2, 2)).to(device)
        ts = torch.zeros((B, 0, 1, 2)).to(device)
        scales = torch.zeros((B, 0, 1, 1)).to(device)
        scores_ransac = torch.zeros((B, 0)).to(device)

        it_matches_ids = []
        dict_corr = {}

        for i_i in range(self.it_matches):

            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except:
                print('[Except Reached]: Invalid matching matrix! ')
                break

            sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = (sampled_idx % num_kpts_grd)

            X = self.metric_coord_sat_B[batch_idx, sampled_idx_sat, :]
            Y = self.metric_coord_grd_B[batch_idx, sampled_idx_grd, :]

            weights = matches_row[batch_idx, sampled_idx]

            dict_corr[i_i] = {'X': X, 'Y': Y, 'weights': weights}

            for kk in range(self.it_RANSAC):

                sampled_idx_ransac = torch.multinomial(weights, self.num_corr_2d_2d)

                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]
                weights_k = weights[batch_idx_ransac, sampled_idx_ransac]

                R, t, scale, ok_rank = weighted_procrustes_2d_with_scale(Y_k, X_k, use_weights=False)

                if not ok_rank:
                    continue

                invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
                invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

                if invalid_t or invalid_R:
                    continue

                score_k = soft_inlier_counting_bev_with_scale(Y, X, R, t, scale, th=self.th_soft_inlier)

                Rs = torch.cat([Rs, R.unsqueeze(1)], 1)
                ts = torch.cat([ts, t.unsqueeze(1)], 1)
                scales = torch.cat([scales, scale.unsqueeze(1)], 1)
                scores_ransac = torch.cat([scores_ransac, score_k], 1)
                it_matches_ids.append(i_i)
                num_valid_h += 1

        if num_valid_h > 0:
            max_ind = torch.argmax(scores_ransac, dim=1)
            R = Rs[batch_idx_ransac[:, 0], max_ind]
            t_metric = ts[batch_idx_ransac[:, 0], max_ind]
            scale = scales[batch_idx_ransac[:, 0], max_ind]
            best_inliers = scores_ransac[batch_idx_ransac[:, 0], max_ind]

            X_best = torch.zeros_like(X)
            Y_best = torch.zeros_like(Y)
            for i_b in range(len(max_ind)):
                X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
            inliers_ref = torch.zeros((B, self.num_samples_matches)).to(device)

            th_ref = self.num_refinements*[self.th_inlier]
            inliers_pre = self.num_corr_2d_2d * torch.ones_like(best_inliers)
            for i_ref in range(len(th_ref)):
                inliers = inlier_counting_bev_with_scale(Y_best, X_best, R, t_metric, scale, th=th_ref[i_ref])

                do_ref = (inliers.sum(-1) >= self.num_corr_2d_2d) * (inliers.sum(-1) > inliers_pre)
                inliers_pre[do_ref] = inliers.sum(-1)[do_ref]

                if (do_ref.sum().float() == 0.).item():
                    break
                inliers_ref[do_ref] = inliers[do_ref]
                R[do_ref], t_metric[do_ref], _, _ = weighted_procrustes_2d_with_scale(Y_best[do_ref], X_best[do_ref],
                                                                     use_weights=True, use_mask=True,
                                                                     check_rank=False,
                                                                     w=inliers_ref[do_ref])
            best_inliers = soft_inlier_counting_bev_with_scale(Y_best, X_best, R, t_metric, scale, th=self.th_inlier)

        else:
            R = torch.zeros((B, 2, 2)).to(matches.device)
            t_metric = torch.zeros((B, 1, 2)).to(matches.device)
            scale = torch.zeros((B, 1, 1)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)

        inliers = None
        if return_inliers:
            if num_valid_h > 0:

                X_best = torch.zeros_like(X)
                Y_best = torch.zeros_like(Y)

                weights_best = torch.zeros_like(weights)
                for i_b in range(len(max_ind)):
                    X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]

                    weights_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['weights'][i_b]

                inliers_idxs = inlier_counting_bev_with_scale(Y_best, X_best, R, t_metric, scale, th=self.th_inlier)
                inliers = []
                for idx_b in range(len(inliers_idxs)):
                    X_inliers = X_best[idx_b, inliers_idxs[idx_b]==1.]
                    Y_inliers = Y_best[idx_b, inliers_idxs[idx_b]==1.]
                    score_inliers = weights_best[idx_b, inliers_idxs[idx_b]==1.]
                    order_corr = torch.argsort(score_inliers, descending=True)
                    inliers_b = torch.cat([X_inliers[order_corr], Y_inliers[order_corr], score_inliers[order_corr].unsqueeze(-1)], dim=1)
                    inliers.append(inliers_b)

        return R, t_metric, scale, best_inliers, inliers

class e2eProbabilisticProcrustesSolver_no_scale():
    def __init__(self, it_RANSAC, it_matches, num_samples_matches, num_corr_2d_2d, num_refinements, th_inlier, th_soft_inlier, metric_coord_sat_B, metric_coord_grd_B):

        self.it_RANSAC = it_RANSAC
        self.it_matches = it_matches
        self.num_samples_matches = num_samples_matches
        self.num_corr_2d_2d = num_corr_2d_2d
        self.num_refinements = num_refinements
        self.th_inlier = th_inlier
        self.th_soft_inlier = th_soft_inlier
        self.metric_coord_sat_B = metric_coord_sat_B
        self.metric_coord_grd_B = metric_coord_grd_B

    def estimate_pose(self, matching_score, return_inliers=False):
        device = matching_score.device
        matches = matching_score.detach()

        B, num_kpts_sat, num_kpts_grd = matches.shape

        matches_row = matches.reshape(B, num_kpts_sat*num_kpts_grd)
        batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, self.num_samples_matches]).reshape(B, self.num_samples_matches)
        batch_idx_ransac = torch.tile(torch.arange(B).view(B, 1), [1, self.num_corr_2d_2d]).reshape(B, self.num_corr_2d_2d)

        num_valid_h = 0
        Rs = torch.zeros((B, 0, 2, 2)).to(device)
        ts = torch.zeros((B, 0, 1, 2)).to(device)
        scores_ransac = torch.zeros((B, 0)).to(device)

        it_matches_ids = []
        dict_corr = {}

        for i_i in range(self.it_matches):

            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except:
                print('[Except Reached]: Invalid matching matrix! ')
                break

            sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = (sampled_idx % num_kpts_grd)

            X = self.metric_coord_sat_B[batch_idx, sampled_idx_sat, :]
            Y = self.metric_coord_grd_B[batch_idx, sampled_idx_grd, :]

            weights = matches_row[batch_idx, sampled_idx]

            dict_corr[i_i] = {'X': X, 'Y': Y, 'weights': weights}

            for kk in range(self.it_RANSAC):

                sampled_idx_ransac = torch.multinomial(weights, self.num_corr_2d_2d)

                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]
                weights_k = weights[batch_idx_ransac, sampled_idx_ransac]

                R, t, ok_rank = weighted_procrustes_2d(Y_k, X_k, use_weights=False)

                if not ok_rank:
                    continue

                invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
                invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

                if invalid_t or invalid_R:
                    continue

                score_k = soft_inlier_counting_bev(X, Y, R, t, th=self.th_soft_inlier)

                Rs = torch.cat([Rs, R.unsqueeze(1)], 1)
                ts = torch.cat([ts, t.unsqueeze(1)], 1)
                scores_ransac = torch.cat([scores_ransac, score_k], 1)
                it_matches_ids.append(i_i)
                num_valid_h += 1

        if num_valid_h > 0:
            max_ind = torch.argmax(scores_ransac, dim=1)
            R = Rs[batch_idx_ransac[:, 0], max_ind]
            t_metric = ts[batch_idx_ransac[:, 0], max_ind]
            best_inliers = scores_ransac[batch_idx_ransac[:, 0], max_ind]

            X_best = torch.zeros_like(X)
            Y_best = torch.zeros_like(Y)
            for i_b in range(len(max_ind)):
                X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
            inliers_ref = torch.zeros((B, self.num_samples_matches)).to(device)

            th_ref = self.num_refinements*[self.th_inlier]
            inliers_pre = self.num_corr_2d_2d * torch.ones_like(best_inliers)
            for i_ref in range(len(th_ref)):
                inliers = inlier_counting_bev(X_best, Y_best, R, t_metric, th=th_ref[i_ref])

                do_ref = (inliers.sum(-1) >= self.num_corr_2d_2d) * (inliers.sum(-1) > inliers_pre)
                inliers_pre[do_ref] = inliers.sum(-1)[do_ref]

                if (do_ref.sum().float() == 0.).item():
                    break
                inliers_ref[do_ref] = inliers[do_ref]
                R[do_ref], t_metric[do_ref], _ = weighted_procrustes_2d(Y_best[do_ref], X_best[do_ref],
                                                                     use_weights=True, use_mask=True,
                                                                     check_rank=False,
                                                                     w=inliers_ref[do_ref])
            best_inliers = soft_inlier_counting_bev(X_best, Y_best, R, t_metric, th=self.th_inlier)

        else:
            R = torch.zeros((B, 2, 2)).to(matches.device)
            t_metric = torch.zeros((B, 1, 2)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)

        inliers = None
        if return_inliers:
            if num_valid_h > 0:

                X_best = torch.zeros_like(X)
                Y_best = torch.zeros_like(Y)
                weights_best = torch.zeros_like(weights)
                for i_b in range(len(max_ind)):
                    X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
                    weights_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['weights'][i_b]

                inliers_idxs = inlier_counting_bev(X_best, Y_best, R, t_metric, th=self.th_inlier)
                inliers = []
                for idx_b in range(len(inliers_idxs)):
                    X_inliers = X_best[idx_b, inliers_idxs[idx_b]==1.]
                    Y_inliers = Y_best[idx_b, inliers_idxs[idx_b]==1.]
                    score_inliers = weights_best[idx_b, inliers_idxs[idx_b]==1.]
                    order_corr = torch.argsort(score_inliers, descending=True)
                    inliers_b = torch.cat([X_inliers[order_corr], Y_inliers[order_corr], score_inliers[order_corr].unsqueeze(-1)], dim=1)
                    inliers.append(inliers_b)

        return R, t_metric, best_inliers, inliers
