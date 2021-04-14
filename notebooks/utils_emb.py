__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from MKLpy import metrics
from MKLpy.preprocessing.kernel_preprocessing import kernel_centering
from joblib import Parallel, delayed
from numba import jit
from sklearn.decomposition import KernelPCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.manifold import TSNE

@jit(nopython=True, parallel=True)
def tanimoto(v1, v2):
    v1_v2 = np.dot(v1, v2)
    v1_sq = np.sum(np.square(v1))
    v2_sq = np.sum(np.square(v2))
    return v1_v2 / (v1_sq + v2_sq - v1_v2)


@jit(nopython=True, parallel=True)
def tanimoto_v2(x1, x2):
    f = (np.linalg.norm(x1, ord=1) + np.linalg.norm(x2, ord=1) - np.linalg.norm(x1 - x2, ord=1))
    g = (np.linalg.norm(x1, ord=1) + np.linalg.norm(x2, ord=1) + np.linalg.norm(x1 - x2, ord=1))
    return f / g


@jit(nopython=True, parallel=True)
def intensional_similarity(v1, v2):
    return np.sum(np.minimum(v1, v2)) / np.sum(np.maximum(v1, v2))


@jit(nopython=True, parallel=True)
def kernel_func(X, Y, ker=None):
    len_x = X.shape[0]
    len_y = Y.shape[0]
    dist_array = np.zeros((len_x, len_y))
    i = 0
    if ker is None:
        kernel = tanimoto
    else:
        kernel = ker
    for a in X:
        a = a.flatten()
        j = 0
        for b in Y:
            b = b.flatten()
            dist = kernel(a, b)
            dist_array[i, j] = dist
            j += 1
        i += 1
    return dist_array


def compute_ker_alignment(X, Y, target, ker_dict, args=None, train=True, n_jobs=-1):
    # Based on Algorithms for Learning Kernels Based on Centered Alignment
    # CORTES et.al
    out_mat = {}
    k_ls = []

    def calc_score(k):
        metric = ker_dict[k]
        if callable(metric):
            X_k = kernel_func(X, Y, ker=metric)
        else:
            if args is not None and k in args:
                X_k = pairwise_kernels(X, Y, metric=metric, **args[k])
            else:
                X_k = pairwise_kernels(X, Y, metric=metric)

        if train:
            X_k_c = kernel_centering(X_k)
            out_mat[k] = metrics.alignment_yy(X_k_c, torch.from_numpy(target))
            k_ls.append(X_k_c.double())
        else:
            k_ls.append(torch.from_numpy(X_k).double())

        return out_mat, k_ls

    r, q = zip(*Parallel(n_jobs=n_jobs, verbose=10)(delayed(calc_score)(k) for k in ker_dict))
    return r, q


def do_emb(X_train, X_test, ker=tanimoto):
    kpca = KernelPCA(kernel=ker, fit_inverse_transform=True)
    X_train_kpca = kpca.fit_transform(X_train)
    X_test_kpca = kpca.transform(X_test)

    X_train_inv = kpca.inverse_transform(X_train_kpca)
    X_test_inv = kpca.inverse_transform(X_test_kpca)

    mse_train = mean_squared_error(X_train, X_train_inv)
    mse_test = mean_squared_error(X_test, X_test_inv)
    print(mse_train)
    print(mse_test)
    return X_train_kpca, X_test_kpca, kpca


def plot_emb_projection(X, y=None, ker=tanimoto_v2, alpha=0.5, params=None, annotate=False,
                        annotate_list=None, label="GO/Pathways", target_name="posOutcome", title="", return_dist=True):
    """
    Plot the row vectors of X and features of X in the same embedded space spanned by PCA Components
    :param X: the data matrix or dataframe
    :param y: the target variable (for labelling)
    :param ker: the kernel function to use
    :param alpha: the exponent to use for matrix factorization
    :return: The pca projects of the row vectors and the columns
    """

    # Do SVD Decomposition
    u, d, v_t = scipy.linalg.svd(X, full_matrices=False)
    d = np.diag(d)
    d_1, d_2 = np.power(d, alpha), np.power(d, 1 - alpha)
    P = u @ d_1
    G = v_t.T @ d_2
    # Apply the kernel on H
    if callable(ker):
        K = kernel_func(G, G, ker=ker)
        # Transform vectors in G to the feature space of K
        K_p = kernel_func(P, G, ker=ker)
        # Apply PCA
        kpca = KernelPCA(kernel="precomputed")
        G_pca = kpca.fit_transform(K)
        P_pca = kpca.transform(K_p)
    else:
        if params is None:
            kpca = KernelPCA(kernel=ker)
        else:
            kpca = KernelPCA(kernel=ker, **params)
        G_pca = kpca.fit_transform(G)
        P_pca = kpca.transform(P)

    print("n_components={0}".format(G_pca.shape))

    # plot the first two components of G_pca and P_pca on the same plot
    markers = {"relapse": ".", "genes": "X", "no_relapse": "+"}
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.grid()

    # Make dataframes for plotting
    G_pca_df = pd.DataFrame(G_pca, index=X.columns)
    P_pca_df = pd.DataFrame(P_pca, index=X.index)
    plot_scatter(G_pca_df, P_pca_df, target_name, ax, label, X.columns.to_list(), annotate, annotate_list, y)
    if return_dist:
        # dist_arr = pairwise_kernels(P_pca, G_pca,metric=ker)
        dist_arr = scipy.spatial.distance_matrix(P_pca, G_pca, p=2)
        dist_df = pd.DataFrame(dist_arr, index=P_pca_df.index, columns=G_pca_df.index)
        return dist_df


def plot_scatter(df1, df2, target_name, ax, label, cols, annotate, annotate_list, y=None):
    if y is not None:
        uniq_vals = y.unique()
        df2 = df2.join(y)
        cmap = colors.ListedColormap(sns.hls_palette(len(uniq_vals)).as_hex())
        for i, v in enumerate(list(uniq_vals)):
            df_i = df2[df2[target_name] == v]
            ax.scatter(df_i[0], df_i[1], c=cmap(i), marker=6, label=str(v))
    else:
        ax.scatter(df2.iloc[:, 0], df2.iloc[:, 1], c='g', marker="+", label="Patients")
    ax.scatter(df1.iloc[:,0], df1.iloc[:,1], c='b', marker="x", label=label)
    if annotate:
        if annotate_list is not None:
            for i in annotate_list:
                x, y = df1.loc[i][0], df1.loc[i][1]
                ax.annotate(i, xy=(x, y), textcoords="offset points")
        else:
            for i in cols:
                x, y = df1.loc[i][0], df1.loc[i][1]
                ax.annotate(i, xy=(x, y), textcoords="offset points")
    ax.axvline(x=0)
    ax.axhline(y=0)
    ax.legend()

## Find optimal kernel parameter using nevergrad by optimizing the kernel alignment

def optimize_ker_param(X, y, param_name, ker="rbf", opt=ng.optimizers.NGOpt, r=(0, 10), budget=100, jobs=2):
    def objective(par):
        param = {param_name: par}
        K = pairwise_kernels(X, metric=ker, **param)
        K = kernel_centering(K)
        return -metrics.alignment_yy(K, torch.from_numpy(y))

    instrum = ng.p.Instrumentation(ng.p.Array(shape=(1,)).set_bounds(r[0], r[1]))
    optimizer = opt(parametrization=instrum, budget=budget, num_workers=jobs)
    recommendation = optimizer.minimize(objective)
    print(recommendation.value)
    print(recommendation.losses)
    return recommendation.value[0][0][0]


def print_array(arr):
    for x in arr:
        sys.stdout.write(x + "\n")
