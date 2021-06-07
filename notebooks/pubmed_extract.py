import sys, time, requests, sqlite3
import xml.etree.ElementTree as ET
from datetime import date
import pickle
import time


def get_pubmed_ids(term, retmax=250, db="pubmed"):
    '''
    Returns a list of pubmed ids for a given search term from the ncbi database
    :param term: the term to search for
    :param retmax: the maximum number of ids to return
    :param db: the databased to search in - default pubmed
    :return: a list of the ids
    '''

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
    url = "{0}db={1}&retmax={2}&term={3}&retmode=xml".format(base_url, db, retmax, term)
    response = requests.get(url).text
    #parse the XML response
    root = ET.fromstring(response)
    id_elms = root.findall("./IdList/Id")
    ids = []
    for elm in id_elms:
        ids.append(elm.text)

    return ids

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
        if abstract is None or u'\N{COPYRIGHT SIGN}' in abstract or abstract == "":
            continue
        collected_abstract[id] = abstract

    return collected_abstract, invalid_ids

def get_go_cat(abstracts):
    """Return the list of GO categories found in an abstract"""

    base_url = "http://candy.hesge.ch/GOCat/result.jsp?cat=ml&json&queryTXT="

    collected_gos = {}
    for k, v in abstracts.items():
        url = "{0}{1}".format(base_url, v)
        response = None
        try:
            response = requests.get(url, headers={"Accept": "application/json"}, timeout=120)
            response = response.json()
            terms = response["all_terms"]
            collected_gos[k] = []
            for term in terms:
                collected_gos[k].append(term["GOid"])
        except Exception as e:
            print(e)
            print(response.text)
            print(k)
            print(v)
        time.sleep(2)
    return collected_gos




if __name__ == '__main__':
    term_1 = "tamoxifen resistance breast cancer"
    term_2 = "ER positive breast cancer biomarker"
    start_time = time.time()
    # pub_ids_1 = get_pubmed_ids(term_1, retmax=200)
    # pub_ids_2 = get_pubmed_ids(term_2, retmax=200)
    # pub_ids_2.extend(pub_ids_1)
    pubmed_id_3 = ['31146164', '33576905', '33781841', '32010961', '33632224', '30326937', '31548378', '33745390', '32200487', '33692831', '33808099', '33747225', '33791842', '32806533', '31165728', '33632871', '33934391', '33717248', '33703991']
    print("Num pubmed articles: %d" % len(pubmed_id_3))
    abstracts, invalid_ids = abstract_download(pubmed_id_3)

    print("Got {} abstracts".format(len(abstracts)))
    print("Invalid IDs list: {}".format(invalid_ids))
    # go_terms = get_go_cat(abstracts)

    path_a = "pubmed_abstracts_pln_pubtator_1.pickle"
    # path_b = "pubmed_gos.pickle"

    with open(path_a, "wb") as fp:
        pickle.dump(abstracts, fp)
    #
    # with open(path_b, "wb") as fp:
    #     pickle.dump(go_terms, fp)


    end_time = time.time()
    print("Total number of seconds: {0}".format(end_time - start_time))

