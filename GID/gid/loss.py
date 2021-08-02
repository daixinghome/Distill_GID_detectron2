import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss

def distill_loss_feature(
    te_features: torch.Tensor,
    st_features: torch.Tensor,
    *,
    loss_mode: str = "L2",
    smooth_l1_beta: float = 0,
) -> torch.Tensor:
    loss_mode = loss_mode.lower()
    assert loss_mode in ["l1", "l2"], loss_mode
    assert te_features.shape == st_features.shape, \
        "Shape of T:{} S:{} features should be the same.".format(te_features.shape, st_features.shape)
    assert te_features.shape == st_features.shape, "T: {}, S:{}".format(te_features, st_features)

    ins_pixel_count = max(1, st_features.shape[0] * st_features.shape[2] * st_features.shape[3])
    if loss_mode == "l1":
        loss = smooth_l1_loss(
            te_features,
            st_features,
            beta=smooth_l1_beta,
            reduction="sum"
        ) / ins_pixel_count
    elif loss_mode == "l2":
        loss = F.mse_loss(
            te_features,
            st_features,
            reduction="sum",
        ) / ins_pixel_count
    else:
        raise NotImplementedError

    return loss


def loss_RkdDistance(
    feature_te: torch.Tensor,
    feature_st: torch.Tensor,
    *,
    dist_mode: str = "L2",
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    assert feature_te.shape[0] == feature_st.shape[0], \
        'Invalid batch nums T:{} S:{} in RKdDistance.'.format(feature_te.shape, feature_st.shape)

    def pdist(e, dist_mode="L2"):
        dist_mode = dist_mode.lower()
        assert dist_mode in ["l1", "l2"], dist_mode

        # 初始维度：(N, C)
        Batch = e.shape[0]
        # Calculate pair-distance
        if dist_mode == "l1":
            res = torch.max(torch.abs(e.reshape(-1, 1) - e.reshape(1, -1)),
                            other=torch.tensor([1e-12]).to(e_square.device)
                           )
        elif dist_mode == "l2":
            # 1. 平方和
            e_square = e.pow(2.0).sum(dim=1)
            # 2. 交叉部分
            prod = torch.matmul(e, e.permute(1, 0))
            # 3. 求结果，保证无0
            res = torch.sqrt(torch.max(e_square.reshape(-1, 1) + e_square.reshape(1, -1) - 2 * prod,
                                       other=torch.tensor([1e-12]).to(e_square.device)
                                      )
                            )
        else:
            raise NotImplementedError

        # 对角线置0,保证下一步操作
        mask_res = torch.ones_like(res).reshape(-1)
        mask_res[::(Batch + 1)] = 0
        mask_res = mask_res.reshape(Batch, Batch)
        res = res * mask_res
        # 只计算非对角线均值
        mask = res > 0
        res = res * mask
        norm_c = res.sum() / torch.max(mask.sum(),
                                       other=torch.tensor([1]).to(e_square.device)
                                      )

        res_norm = res / torch.max(norm_c,
                                   other=torch.tensor([1e-12]).to(e_square.device)
                                  )
        return res_norm

    if feature_st.shape[0] < 3:
        loss = torch.tensor([0.]).to(feature_st.device)
    else:
        feature_te = feature_te.reshape(feature_te.shape[0], -1)
        feature_st = feature_st.reshape(feature_st.shape[0], -1)

        t_d = pdist(feature_te, dist_mode)
        s_d = pdist(feature_st, dist_mode)

        off_diagonal_num = (feature_te.shape[0] * feature_te.shape[0] - feature_te.shape[0])
        loss = smooth_l1_loss(t_d, s_d, beta=smooth_l1_beta, reduction="sum") / off_diagonal_num

    return loss


def distill_loss_relation(
    te_features: torch.Tensor,
    st_features: torch.Tensor,
    *,
    loss_mode: str = "IRKD",
    dist_mode: str = "L2",
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    loss_mode = loss_mode.lower()
    assert loss_mode in ["irkd"], loss_mode
    assert te_features.shape == st_features.shape, "T: {}, S:{}".format(te_features, st_features)

    if loss_mode == "irkd":
        loss = loss_RkdDistance(
            te_features,
            st_features,
            dist_mode=dist_mode,
            smooth_l1_beta=smooth_l1_beta
        )
    else:
        raise NotImplementedError

    return loss


def distill_loss_response(
    te_features: torch.Tensor,
    st_features: torch.Tensor,
    distill_mask: torch.Tensor,
    *,
    loss_mode: str = "L2",
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    loss_mode = loss_mode.lower()
    assert loss_mode in ["bce", "l1", "l2"], loss_mode
    assert te_features.shape == st_features.shape, "T: {}, S:{}".format(te_features, st_features)

    valid_nums = distill_mask.sum()
    st_features = st_features[distill_mask]
    te_features = te_features[distill_mask]
    if loss_mode == "bce":
        loss = F.binary_cross_entropy_with_logits(
            st_features,
            te_features.sigmoid(),
            reduction="sum",
        ) / max(1, valid_nums)
    elif loss_mode == "l1":
        loss = smooth_l1_loss(
            st_features,
            te_features,
            beta=smooth_l1_beta,
            reduction="sum",
        ) / max(1, valid_nums)
    elif loss_mode == "l2":
        loss = F.mse_loss(
            st_features,
            te_features,
            reduction="sum",
        ) / max(1, valid_nums)
    else:
        raise NotImplementedError

    return loss
