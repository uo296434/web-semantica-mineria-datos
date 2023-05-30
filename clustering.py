import sys
import spacy
import random
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import OrderedDict

with open('noticias_negacionistas_segmentadas.json', 'r', encoding='utf-8') as json_file:
    noticias_negacionistas_segmentadas = json.load(json_file)

with open('salida_clustering.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f

    print("CLUSTERING CON K-MEANS\n\n")
    # Clustering con k-means
    nlp = spacy.load("es_core_news_sm")
    stop_words = list(spacy.lang.es.stop_words.STOP_WORDS)
    vectorizador = TfidfVectorizer(encoding="utf-8", lowercase=True,
                                stop_words=stop_words, ngram_range=(1,3), 
                                max_features=10000)
    doc_term_matrix = vectorizador.fit_transform(noticias_negacionistas_segmentadas)

    num_clusters = 25
    clustering = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=1000, n_init=1, verbose=True)
    clustering.fit(doc_term_matrix)

    clustered_docs = dict()
    docs_per_cluster = dict()

    for i in range(len(clustering.labels_)):
        try:
            clustered_docs[clustering.labels_[i]].append(i)
            docs_per_cluster[clustering.labels_[i]]+=1
        except:
            clustered_docs[clustering.labels_[i]] = list()
            clustered_docs[clustering.labels_[i]].append(i)
            docs_per_cluster[clustering.labels_[i]]=1

    ids = list(docs_per_cluster.keys())
    ids.sort()
    sorted_docs_per_cluster = {i: docs_per_cluster[i] for i in ids}

    terminos = vectorizador.get_feature_names_out()

    indice_cluster_terminos = clustering.cluster_centers_.argsort()[:, ::-1]

    for cluster_id in sorted_docs_per_cluster:

        print()
        print("Cluster %d (%d documentos): " % (cluster_id, docs_per_cluster[cluster_id]), end="")

        for term_id in indice_cluster_terminos[cluster_id, :10]:
            if clustering.cluster_centers_[cluster_id][term_id]!=0:
                print('"%s"' % terminos[term_id], end=" ")
        
        print()

        ejemplares = clustered_docs[cluster_id]
        random.shuffle(ejemplares)
        for ejemplar in ejemplares[0:8]:
            print("\t",noticias_negacionistas_segmentadas[ejemplar][0:250],"...")
        
        print()

with open('clustered_docs.json', 'w', encoding='utf-8') as json_file:
    clustered_docs_int_keys = {int(k): v for k, v in clustered_docs.items()}
    json.dump(clustered_docs_int_keys, json_file)