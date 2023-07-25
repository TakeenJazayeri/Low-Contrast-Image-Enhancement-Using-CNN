import numpy as np
from math import sqrt
import cv2, random, math

def GcsDecolor2 (im, Lpp = 0.25):
    (n, m, _) = im.shape
    W = wei()

    new_row = round(64/sqrt(n*m)*n)
    new_col = round(64/sqrt(n*m)*m)

    ims = cv2.resize(im, (new_col, new_row))
    (B, G, R) = cv2.split(ims)
    imV = np.stack((R.flatten(), G.flatten(), B.flatten()), axis=1)

    imV_len = imV.shape[0]
    t1 = permutation(imV_len)
    Pg = np.array([imV[i] - imV[t1[i]] for i in range(imV_len)])

    ims = cv2.resize(im, (round(new_col/2), round(new_row/2)))
    (Bs, Gs, Rs) = cv2.split(ims)
    Rx = (Rs[:, :-1] - Rs[:, 1:]).flatten()
    Gx = (Gs[:, :-1] - Gs[:, 1:]).flatten()
    Bx = (Bs[:, :-1] - Bs[:, 1:]).flatten()

    Ry = (Rs[:-1, :] - Rs[1:, :]).flatten()
    Gy = (Gs[:-1, :] - Gs[1:, :]).flatten()
    By = (Bs[:-1, :] - Bs[1:, :]).flatten()

    Pl = np.stack((np.concatenate((Rx, Ry)), np.concatenate((Gx, Gy)), np.concatenate((Bx, By))), axis=1)
    P = np.concatenate((Pg, Pl))

    len_P = P.shape[0]
    for i in range(len_P-1, -1, -1):
        if (math.sqrt(sum(P[i] ** 2)) / 1.41) < 0.05:
            P = np.delete(P, i, 0)

    L = np.dot(P, np.transpose(W))
    LL = np.stack((L, L, L), axis = 2)


    len_W = W.shape[0]
    P0 = np.transpose(np.array([abs(P[:, 0])]))
    P1 = np.transpose(np.array([abs(P[:, 1])]))
    P2 = np.transpose(np.array([abs(P[:, 2])]))

    LL3_0 = np.append(P0, P0, axis=1)
    LL3_1 = np.append(P1, P1, axis=1)
    LL3_2 = np.append(P2, P2, axis=1)

    for i in range(len_W-2):
        LL3_0 = np.append(LL3_0, P0, axis=1)
        LL3_1 = np.append(LL3_1, P1, axis=1)
        LL3_2 = np.append(LL3_2, P2, axis=1)

    LL3 = np.stack((LL3_0+Lpp, LL3_1+Lpp, LL3_2+Lpp), axis = 2)

    U = (abs(LL)*LL3)/(LL**2 + LL3**2)

    Es = np.mean(np.mean(U, axis = 0), axis = 1)
    bw = sum(i for i in range(len(Es)) if Es[i] == np.max(Es))

    b,g,r = cv2.split(im)
    dst = cv2.addWeighted(cv2.addWeighted(r, W[bw,0], g, W[bw,1], 0.0), W[bw,0]+W[bw,1], b, W[bw,2], 0.0)

    return dst


def permutation (n):
    _1_to_n = [_ for _ in range(n)]
    return [_1_to_n.pop(random.randrange(n-i)) for i in range(n)]


def wei ():
    return  np.array([
        [0, 0, 1.0],
        [0, 0.1, 0.9],
        [0, 0.2, 0.8],
        [0, 0.3, 0.7],
        [0, 0.4, 0.6],
        [0, 0.5, 0.5],
        [0, 0.6, 0.4],
        [0, 0.7, 0.3],
        [0, 0.8, 0.2],
        [0, 0.9, 0.1],
        [0, 1.0, 0],
        [0.1, 0, 0.9],
        [0.1, 0.1, 0.8],
        [0.1, 0.2, 0.7],
        [0.1, 0.3, 0.6],
        [0.1, 0.4, 0.5],
        [0.1, 0.5, 0.4],
        [0.1, 0.6, 0.3],
        [0.1, 0.7, 0.2],
        [0.1, 0.8, 0.1],
        [0.1, 0.9, 0],
        [0.2, 0 , 0.8],
        [0.2, 0.1, 0.7],
        [0.2, 0.2, 0.6],
        [0.2, 0.3, 0.5],
        [0.2, 0.4, 0.4],
        [0.2, 0.5, 0.3],
        [0.2, 0.6, 0.2],
        [0.2, 0.7, 0.1],
        [0.2, 0.8, 0],
        [0.3, 0, 0.7],
        [0.3, 0.1, 0.6],
        [0.3, 0.2, 0.5],
        [0.3, 0.3, 0.4],
        [0.3, 0.4, 0.3],
        [0.3, 0.5, 0.2],
        [0.3, 0.6, 0.1],
        [0.3, 0.7, 0.0],
        [0.4, 0  , 0.6],
        [0.4, 0.1, 0.5],
        [0.4, 0.2, 0.4],
        [0.4, 0.3, 0.3],
        [0.4, 0.4, 0.2],
        [0.4, 0.5, 0.1],
        [0.4, 0.6, 0.0],
        [0.5, 0  , 0.5],
        [0.5, 0.1, 0.4],
        [0.5, 0.2, 0.3],
        [0.5, 0.3, 0.2],
        [0.5, 0.4, 0.1],
        [0.5, 0.5, 0],
        [0.6, 0, 0.4],
        [0.6, 0.1, 0.3],
        [0.6, 0.2, 0.2],
        [0.6, 0.3, 0.1],
        [0.6, 0.4, 0.0],
        [0.7, 0, 0.3],
        [0.7, 0.1, 0.2],
        [0.7, 0.2, 0.1],
        [0.7, 0.3, 0.0],
        [0.8, 0, 0.2],
        [0.8, 0.1, 0.1],
        [0.8, 0.2, 0.0],
        [0.9, 0, 0.1],
        [0.9, 0.1, 0.0],
        [1.0, 0, 0]
    ])