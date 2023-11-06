import mindspore as ms
import mindspore.ops as ops
import numpy as np



def euclidean_distance(input1, input2):
    """Computes euclidean squared distance.
    Args:
        input1: 2-D feature matrix.
        input2: 2-D feature matrix.
    Returns:
        distance matrix.
    """
    m, n = input1.shape[0], input2.shape[0]

    shape_tensor1 = ms.Tensor(np.zeros((m, n), dtype=np.float32))
    shape_tensor2 = ms.Tensor(np.zeros((n, m), dtype=np.float32))
    op_pow = ops.Pow()

    mat1 = op_pow(input1, 2).sum(
        axis=1, keepdims=True).expand_as(shape_tensor1)
    mat2 = op_pow(input2, 2).sum(
        axis=1, keepdims=True).expand_as(shape_tensor2).T
    distmat = mat1 + mat2
    matmul = ops.MatMul(False, True)
    cast = ops.Cast()
    input1 = cast(input1, ms.float32)
    input2 = cast(input2, ms.float32)
    output = cast(matmul(input1, input2), ms.float32)
    distmat = distmat - 2 * output

    return distmat.asnumpy()




# mINP
def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        # order[2] = 0
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)


        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)


        raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches

        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        pid = tuple(pid.asnumpy())
        camid = tuple(camid.asnumpy())
        # self.feats.append(feat.cpu())
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch
        feats = ops.cat(self.feats, axis=0)
        if self.feat_norm:
            print("The test feature is normalized")

            l2_normalize = ops.L2Normalize(axis=1)
            feats = l2_normalize(feats)


        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('Not available now.')
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        cmc,mAP,minp = eval_market1501(distmat,q_pids,g_pids,q_camids,g_camids,max_rank=50)
        mAP = np.mean(mAP)

        minp = np.mean(minp)
        return cmc, mAP,minp, distmat, self.pids, self.camids, qf, gf


