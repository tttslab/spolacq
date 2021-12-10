import random

import torch


def random_index(ban_i, ceil_i):
    max_i = ceil_i - 1
    assert 0 <= ban_i <= max_i
    i = random.randint(0, max_i - 1)
    if i == ban_i:
        i = max_i
    return i


def sim_loss_batchsum(feat_vs, feat_as, rho = 4.0, eps = 1e-5):
    ret = 0

    for i in range(len(feat_vs)):
        j = random_index(i, len(feat_vs))
        ret += sim_loss(feat_vs[i], feat_as[i], feat_vs[j], feat_as[j], rho, eps)
    return ret


def sim_loss(feat_v, feat_a, feat_v_imp,feat_a_imp, rho = 4.0, eps=1e-5):
    assert len(feat_v) == len(feat_a)

    score_corr = ((feat_v - feat_a)**2).sum()**(1/2)
    score_impa = ((feat_v - feat_a_imp)**2).sum()**(1/2)
    if score_corr.is_cuda:
        zeros = torch.cuda.FloatTensor((1,)).fill_(0)
    else:
        zeros = torch.zeros(*score_corr.shape)
    ret = 0.5*score_corr**2 + 0.5*torch.max(zeros, rho - score_impa)**2
    ret = torch.mean(ret)
    return ret