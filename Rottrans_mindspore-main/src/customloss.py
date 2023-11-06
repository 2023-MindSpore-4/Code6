import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    op = ops.LpNorm(axis=axis, p=2, keepdim=True)
    output = op(x)
    x = 1. * x / (output.expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.shape[0], y.shape[0]

    pow = ops.Pow()
    broadcast_to_xx = ops.BroadcastTo((m,n))
    broadcast_to_yy = ops.BroadcastTo((n,m))
    xx = broadcast_to_xx(pow(x, 2).sum(1, keepdims=True))
    yy = broadcast_to_yy(pow(y, 2).sum(1, keepdims=True)).t()

    dist = xx + yy
    dist = dist - 2 * ops.matmul(x, y.t())

    dist = ops.clamp(dist, min=1e-12).sqrt()
    return dist



def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.shape[0], y.shape[0]


    pow = ops.Pow()
    broadcast_to_x_norm = ops.BroadcastTo((m,n))
    broadcast_to_y_norm = ops.BroadcastTo((n,m))

    x_norm = broadcast_to_x_norm(pow(x, 2).sum(1, keepdims=True).sqrt())
    y_norm = broadcast_to_y_norm(pow(y, 2).sum(1, keepdims=True).sqrt()).t()


    xy_intersection = ops.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels):
    """ Search min negative and max positive distances
    """
    def get_max(dist_mat_, idxs, inv=False):
        dist_mat_ = dist_mat_.copy()
        if inv:
            dist_mat_ = -dist_mat_
        dist_mat_[~idxs] = dist_mat_.min() - 1
        pos_max = dist_mat_.argmax(axis=-1)
        maxes = dist_mat_.take(pos_max, axis=-1).diagonal()
        return pos_max, maxes

    n = dist_mat.shape[0]

    labels_sq = ops.expand_dims(labels, -1).repeat(n, axis=-1)

    is_pos = ops.equal(labels_sq, labels_sq.T)
    is_neg = ops.not_equal(labels_sq, labels_sq.T)

    p_inds, dist_ap = get_max(dist_mat, is_pos)
    n_inds, dist_an = get_max(dist_mat, is_neg, inv=True)

    return dist_ap, -dist_an






class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """
    def __init__(self, margin=None, hard_factor=0.0, reduction='none'):
        self.margin = margin
        self.hard_factor = hard_factor
        self.ones_like = ops.OnesLike()
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction='none')

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = self.ones_like(dist_an)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            # there's something wrong with nn.SoftMarginLoss(reduction='mean'), it creates WARNINGs
            loss = self.ranking_loss(dist_an - dist_ap, y)
            loss = ms.Tensor.mean(loss)
        return loss, dist_ap, dist_an





class CustomLoss(nn.Cell):
    """
    CustomLoss: TRI_LOSS + L1_LOSS_ID + L1_LOSS_TR
    """
    def __init__(self, cfg, margin=None, hard_factor=0.0):
        super(CustomLoss, self).__init__()
        self.DATALOADER_SAMPLER = cfg.DATALOADER.SAMPLER
        self.MODEL_METRIC_LOSS_TYPE = cfg.MODEL.METRIC_LOSS_TYPE
        self.MODEL_IF_LABELSMOOTH = cfg.MODEL.IF_LABELSMOOTH
        self.MODEL_ID_LOSS_WEIGHT = cfg.MODEL.ID_LOSS_WEIGHT
        self.MODEL_TRIPLET_LOSS_WEIGHT = cfg.MODEL.TRIPLET_LOSS_WEIGHT
        self.SOLVER_MARGIN = cfg.SOLVER.MARGIN
        

        self.margin = margin
        self.hard_factor = hard_factor
        self.l1loss = nn.SmoothL1Loss(reduction='mean')

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
            if cfg.MODEL.NO_MARGIN:
                self.triplet = TripletLoss()
            else:
                self.triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        else:
            print('expected METRIC_LOSS_TYPE should be triplet but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))


    def construct(self, logits, labels):
        """compute loss"""

        score, feat = logits
        target = labels

        if self.DATALOADER_SAMPLER == 'softmax_triplet':
            if self.MODEL_METRIC_LOSS_TYPE == 'triplet':
                if self.MODEL_IF_LABELSMOOTH == 'on':
                    print('expected MODEL.IF_LABELSMOOTH should be off now but got {}'.format(self.MODEL_IF_LABELSMOOTH))
                else:

                    if isinstance(score, list):
                        ID_LOSS = [ops.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 *ID_LOSS + 0.5 *ops.cross_entropy(score[0], target)

                        mean = [scor for scor in score[1:]]
                        mean = sum(mean) / len(mean)
                        L1_LOSS_ID = self.l1loss(score[0], mean)
                    else:
                        ID_LOSS = ops.cross_entropy(score, target)


                    if isinstance(feat, list):
                        TRI_LOSS = [self.triplet(feats, target)[0] for feats in feat[1:]]
                        TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                        TRI_LOSS = TRI_LOSS + self.triplet(feat[0], target)[0]
                        mean = [feats for feats in feat[1:]]
                        mean = sum(mean) / len(mean)
                        L1_LOSS_TR = self.l1loss(feat[0], mean)
                    else:
                        TRI_LOSS = self.triplet(feat, target)[0]
                    CUSTOM_LOSS = self.MODEL_ID_LOSS_WEIGHT * ID_LOSS + self.MODEL_TRIPLET_LOSS_WEIGHT * TRI_LOSS + L1_LOSS_ID + L1_LOSS_TR
                    return CUSTOM_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet but got {}'.format(self.MODEL_METRIC_LOSS_TYPE))
        else:
            print('expected sampler should be softmax_triplet but got {}'.format(self.DATALOADER_SAMPLER))