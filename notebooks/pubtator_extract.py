__author__ = 'Abdulrahman Semrie<hsamireh@gmail.com>'
import sys, time, requests, sqlite3
import xml.etree.ElementTree as ET
from datetime import date
import pickle
import time
from pubmed_extract import get_pubmed_ids


def get_pub_ann(pub_ids, n=5):

    ##Make 100 requests at a time due to api restriction
    base_url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?concepts=gene&pmids="
    k, m = divmod(len(pub_ids), n)
    ids_lst = (pub_ids[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    collected_ann = {}

    for ids in ids_lst:
        url = base_url + ",".join(ids)
        response = requests.get(url)
        response = response.text
        root = ET.fromstring(response)
        docs = root.findall("./document")

        for doc in docs:
            pubmed_id_el = doc.findall("./id")
            pubmed_id = pubmed_id_el[0].text
            collected_ann[pubmed_id] = []

            gene_els = doc.findall("./passage/annotation/infon")

            for gene_el in gene_els:
                if gene_el.attrib["key"] == "identifier":
                    entrez_ids = gene_el.text.split(";")
                    if "None" in entrez_ids: # Interestingly there some genes don't have entrez id and it returns 'None'. Check pubmed id 33777781 for example
                        entrez_ids.remove("None")

                    collected_ann[pubmed_id].extend(entrez_ids)

            if len(collected_ann[pubmed_id]):
                collected_ann[pubmed_id] = set([int(x) for x in collected_ann[pubmed_id]])
            else:
                del collected_ann[pubmed_id]


    return collected_ann



if __name__ == '__main__':
    term_1 = "tamoxifen resistance breast cancer"
    term_2 = "ER positive breast cancer biomarker"
    term_3 = "breast cancer pathways"
    start_time = time.time()
    pub_ids_1 = get_pubmed_ids(term_1, retmax=500)
    pub_ids_2 = get_pubmed_ids(term_2, retmax=500)
    pub_ids_3 = get_pubmed_ids(term_3, retmax=500)
    pub_ids_2.extend(pub_ids_1)
    pub_ids_3.extend(pub_ids_2)
    print("Num pubmed articles: %d" % len(pub_ids_3))
    gene_annotation = get_pub_ann(pub_ids_3, n=50)

    path_a = "pubmed_gene_ann.pickle"

    with open(path_a, "wb") as fp:
        pickle.dump(gene_annotation, fp)

    end_time = time.time()
    print("Total number of seconds: {0}".format(end_time - start_time))