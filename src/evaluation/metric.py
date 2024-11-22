import numpy as np
from collections import defaultdict

def ranking_metrics_at_k(liked_items, scores, reco_items):
    """
    Calculates ndcg, average precision (aP) for `one user`.
    If you want to derive ndcg, map for n users, you should average them over n.

    liked_items: item ids selected by one user.
    scores: `relevance` associated with liked_items. Could be ratings or indicator value depending on target y.
    If target y is 1 or 0, scores would be np.array([1,1,1]). If target y is ratings, scores would be np.array([3,5,4.5])
    reco_items: item ids recommended for one user.
    """
    assert liked_items.shape == scores.shape

    # number of recommended items
    K = len(reco_items)
    # in case user liked items less than K
    K = min(len(liked_items), K)

    # sort liked_items by descending scores
    # if scores are indicator values, sorted array would be [1,1,1,0,0,0]
    idx = np.argsort(scores)[::-1]
    scores = scores[idx]
    liked_items = liked_items[idx]

    # ap
    ap = 0
    # ndcg
    dcg = (scores / np.log2(np.arange(2, K + 2)))
    idcg = np.sum(dcg)
    ndcg = 0
    hit = 0

    likes = defaultdict(bool)
    for item in liked_items:
        likes[item] = True

    for i in range(K):
        if likes[reco_items[i]] == 1:
            hit += 1
            ap += hit / (i + 1)
            ndcg += dcg[i] / idcg
    ap /= K

    return {
        "ap": ap,
        "ndcg": ndcg
    }