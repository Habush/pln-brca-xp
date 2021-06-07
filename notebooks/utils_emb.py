__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'

import sys
import urllib
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import pandas as pd
import requests
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
from goatools.cli.find_enrichment import GoeaCliFnc
import collections as cx
from convert_symbol_to_entrez import convert_symbol_to_geneid
import subprocess
from IPython.display import Image
import xml.etree.ElementTree as ET
from goatools.semantic import *

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


def print_array(arr, fp=sys.stdout):
    for x in arr:
        fp.write(str(x) + "\n")


def generate_over_under_expr(expr_df, summary_df, target):
    log_pos_genes = summary_df[summary_df["log2fc"] > 0]["gene"].to_list()
    log_neg_genes = summary_df[summary_df["log2fc"] < 0]["gene"].to_list()
    case_idx = target[target == 0].index
    ctr_idx = target[target == 1].index

    genes = summary_df["gene"].to_list()
    filtered_expr_df = expr_df[genes]
    d1 = expr_df.shape[0]
    overexpr_1_col_names, overexpr_0_col_names = {}, {}
    underexpr_col_1_names, underexpr_col_0_names = {}, {}

    def overexpr_1_series(col):
        name = col.name + "_overexpr"
        overexpr_1_col_names[col.name] = name
        arr = np.zeros((d1,))
        for i, idx in enumerate(col.index):
            if idx in ctr_idx:
                arr[i] = col.iloc[i]
        return pd.Series(arr, name=name, index=col.index)

    def overexpr_0_series(col):
        name = col.name + "_overexpr"
        overexpr_0_col_names[col.name] = name
        arr = np.zeros((d1,))
        for i, idx in enumerate(col.index):
            if idx in case_idx:
                arr[i] = col.iloc[i]
        return pd.Series(arr, name=name, index=col.index)

    def underexpr_1_series(col):
        name = col.name + "_underexpr"
        underexpr_col_1_names[col.name] = name
        arr = np.zeros((d1,))
        for i, idx in enumerate(col.index):
            if idx in ctr_idx:
                arr[i] = col.iloc[i]
        return pd.Series(arr, name=name, index=col.index)

    def underexpr_0_series(col):
        name = col.name + "_underexpr"
        underexpr_col_0_names[col.name] = name
        arr = np.zeros((d1,))
        for i, idx in enumerate(col.index):
            if idx in case_idx:
                arr[i] = col.iloc[i]
        return pd.Series(arr, name=name, index=col.index)

    log_pos_df = filtered_expr_df[log_pos_genes]
    log_neg_df = filtered_expr_df[log_neg_genes]

    overexpr_1_df, underexpr_1_df = log_pos_df.apply(overexpr_1_series), log_neg_df.apply(underexpr_1_series)
    overexpr_0_df, underexpr_0_df = log_neg_df.apply(overexpr_0_series), log_pos_df.apply(underexpr_0_series)

    overexpr_1_df, underexpr_1_df = overexpr_1_df.rename(columns=overexpr_1_col_names), underexpr_1_df.rename(columns=underexpr_col_1_names)
    overexpr_0_df, underexpr_0_df = overexpr_0_df.rename(columns=overexpr_0_col_names), underexpr_0_df.rename(columns=underexpr_col_0_names)

    df_out = pd.concat([ overexpr_1_df, underexpr_1_df, overexpr_0_df, underexpr_0_df], axis=1, join="inner", verify_integrity=True)
    return df_out

def request_gos(go_list):
    url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"
    go_terms = ""
    for i, g in enumerate(go_list):
        if i == len(go_list) - 1:
            go_terms += g
        else:
            go_terms += g + ","


    go_terms = urllib.parse.quote(go_terms)
    url += go_terms
    res = requests.get(url, headers={"Accept" : "application/json"})
    res = res.json()
    return res["results"]

def filter_gos_ns(go_lst, namespace):
    res = request_gos(go_lst[:5])
    gos = []
    for r in res["results"]:
        if r["aspect"] == namespace:
            gos.append(r)

    return gos

def create_go_df(go_lst):
    res = request_gos(go_lst)
    go_dict = {"ID": [], "Name": []}
    for r in res:
        go_dict["ID"].append(r["id"])
        go_dict["Name"].append(r["name"])

    df = pd.DataFrame.from_dict(go_dict)
    return df

def run_gene_enrich(study_lst, pop_lst, path, sym2geneid, ns='BP,MF,CC', convert_study=True, convert_pop=True, p_val=0.05):

    if convert_study: study_ids = convert_symbol_to_geneid(study_lst, sym2geneid)
    else: study_ids = study_lst
    if convert_pop: pop_ids = convert_symbol_to_geneid(pop_lst, sym2geneid)
    else: pop_ids = pop_lst

    study_path, pop_path = os.path.join(path, "study_ids"), os.path.join(path, "pop_ids")
    with open(study_path, "w") as fp:
        print_array(study_ids, fp)

    with open(pop_path, "w") as fp:
        print_array(pop_ids, fp)

    opt = {
        'annofmt': None,
        'alpha' : 0.05,
        'compare' : False,
        'filenames' : [study_path,
                        pop_path, 'gene2go'],
        'goslim' : 'datasets/goslim_generic.obo',
        'indent' : False,
        'method' : 'bonferroni,sidak,holm,fdr_bh',
        # 'method': 'fdr_bh',
        'min_overlap' : 0.7,
        'no_propagate_counts' : False,
        'obo' : 'datasets/go-basic.obo',
        # 'outfile' : 'datasets/goea_tx_bp.txt',
        'outfile_detail' : None,
        'ns': ns,
        'pval' : p_val,
        'pval_field' : 'uncorrected',
        'pvalcalc' : 'fisher',
        'ratio' : None,
        'relationship': True,
        'relationships': None,
        'sections' : None,
        'ev_inc': None,
        'ev_exc': None,
        'taxid': 9606
        # BROAD 'remove_goids': None,
        }
    args = cx.namedtuple("Namespace", " ".join(opt.keys()))
    args = args(**opt)
    goea = GoeaCliFnc(args)
    res = goea.get_results_sig()
    res_path = os.path.join(path, "go_ge_{0}.tsv".format(ns))
    goea.prt_outfiles_flat(res, [res_path])
    return pd.read_table(res_path)

def find_overlap_go(lst1, lst2):
    lst1 = [x for x in lst1 if x.startswith("GO:")]
    lst2 = [x for x in lst2 if x.startswith("GO:")]
    overlap = list(set(lst1) & set(lst2))
    print("Num overlap: " + str(len(overlap)))
    diff_1 = list(set(lst1) - set(lst2))
    print("Num found in list 1, not in list 2:" + str(len(diff_1)))
    diff_2 = list(set(lst2) - set(lst1))
    print("Num found in list 2, not in list 1: " + str(len(diff_2)))
    return overlap, diff_1, diff_2


def draw_gos(lst1, lst2, color_1="#3bd163", color_2="#7276e0"):
    arg = "/home/xabush/venv/bin/go_plot.py  "
    for i in lst1:
        arg += "{}{} ".format(i, color_1)

    for i in lst2:
        arg += "{}{} ".format(i, color_2)

    arg += "-o datasets/aaa_lin.png --gaf=datasets/goa_human.gaf"
    print(arg)
    cp = subprocess.run([arg], capture_output=True, shell=True, check=True)
    print(cp.stdout)
    return Image(filename="datasets/aaa_lin.png")

def semantic_sim_matrix(lst_1, lst_2, godag, termcounts):
    mat = np.zeros(shape=(len(lst_1), len(lst_2)))

    for i, go_1 in enumerate(lst_1):
        for j, go_2 in enumerate(lst_2):
            mat[i, j] = lin_sim(go_1, go_2, godag, termcounts)

    return mat


def filter_similar_gos(lst_1, lst_2, godag, termcounts, score=0.4):
    idx = []
    for i, go_1 in enumerate(lst_1):
        for j, go_2 in enumerate(lst_2):
            sim = lin_sim(go_1, go_2, godag, termcounts)
            if sim > score:
                idx.append(i)
                break
    res = lst_1
    for i in reversed(idx):
        del res[i]
    return res

def abstract_download(pmids):
    """
        This method returns abstract for a given pmid and add to the abstract data
    """

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&retmode=xml&rettype=abstract"
    collected_abstract = {}

    ids_str_ls = ','.join(pmids)
    url = "{0}&id={1}".format(base_url, ids_str_ls)
    response = requests.get(url)
    response = response.text
    root = ET.fromstring(response)

    articles = root.findall("./PubmedArticle")
    pubmed_ids = []
    abstracts = []
    invalid_ids = []
    for article in articles:
        pubmed_elms = article.findall("./MedlineCitation/PMID")
        abstract_elms = article.findall('./MedlineCitation/Article/Abstract/AbstractText')

        pubmed_id = pubmed_elms[0].text
        abstract = None
        try:
            if len(abstract_elms) > 0:
                if len(abstract_elms) > 1 and "Label" in abstract_elms[0].attrib:
                    abstract_elm = [x.text for x in abstract_elms if x.attrib["Label"].upper() == "CONCLUSION" or x.attrib["Label"].upper() == "CONCLUSIONS"]
                    if len(abstract_elm) > 0: abstract = abstract_elm[0]
                    else:
                        invalid_ids.append(pubmed_id)
                else:
                    abstract = abstract_elms[0].text

                pubmed_ids.append(pubmed_id)
                abstracts.append(abstract)
            else:
                invalid_ids.append(pubmed_id)
        except Exception as e:
            print(e)
            print(pubmed_id)
            sys.exit(1)
    assert len(pubmed_ids) == len(abstracts)



    for id, abstract in zip(pubmed_ids, abstracts):
        collected_abstract[id] = abstract

    return collected_abstract, invalid_ids