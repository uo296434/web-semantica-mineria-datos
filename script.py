# -*- coding: utf-8 -*-

import json
from simhash import Simhash, SimhashIndex
from tqdm import tqdm 
from collections import Counter
import random
import spacy
from nltk.tokenize import TextTilingTokenizer
from nltk.corpus import stopwords
import sys

original_stdout = sys.stdout

# Primera parte (noticias negacionistas)

with open('salida_negacionistas.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f


    # Obtención noticias negacionistas y noticias no negacionistas

    noticias_negacionistas = list()
    noticias_no_negacionistas = list()

    campos = ["headline","alternativeHeadline","articleBody"]

    with open("noticias-cambio-climático-españa-abril-2022-abril-2023-finales-sentencias-disagree-NOT-disagree.ndjson", "r", encoding="utf-8") as f:
        lineas = f.readlines()

        for linea in lineas:
            try:
                data = json.loads(linea)

                texto = " ".join(data["sentencias"])
                texto = "".join(texto.splitlines())

                num_disagree = len(data["sentencias_disagree"])
                num_NOT_disagree = len(data["sentencias_NOT_disagree"])

                if num_disagree>num_NOT_disagree:
                    noticias_negacionistas.append(texto)
                else:
                    noticias_no_negacionistas.append(texto)
            except Exception as e:
                pass

    print(f"Número de noticias negacionistas: {len(noticias_negacionistas)}")


    # Eliminación de noticias negacionista cuasi-duplicadas

    firmas = []
    valor_f = 128
    observaciones = []

    with tqdm(total=len(noticias_negacionistas)) as barra:
        for i in range(len(noticias_negacionistas)):
            texto = noticias_negacionistas[i]
            firma = Simhash(texto, f=valor_f)
            firmas.append((i,firma))    
            barra.update(1)

    indice = SimhashIndex(firmas, k=10, f=valor_f)

    with tqdm(total=len(noticias_negacionistas)) as barra:
        for i in range(len(noticias_negacionistas)):
            firma = firmas[i][1]
            duplicados = indice.get_near_dups(firma)
            observaciones.append(len(duplicados))
            barra.update(1)

    noticias_negacionistas_no_duplicadas = list()
    ignorar = list()

    with tqdm(total=len(noticias_negacionistas)) as barra:
        for i in range(len(noticias_negacionistas)):
            if i not in ignorar:
                texto = noticias_negacionistas[i]
                firma = firmas[i][1]
                duplicados = indice.get_near_dups(firma)

                if len(duplicados)==1:
                    noticias_negacionistas_no_duplicadas.append(texto)
                    ignorar.append(i)
                else:
                    random.shuffle(duplicados)
                    ident = int(duplicados[0])        
                    noticias_negacionistas_no_duplicadas.append(noticias_negacionistas[ident])

                    for ident in duplicados:
                        ignorar.append(int(ident))

            barra.update(1)

    print()
    print(f"Noticias negacionistas tras eliminar las cuasi-duplicadas: {len(noticias_negacionistas_no_duplicadas)}")
    print()
    print()
    print("Segmentación de las noticias negacionistas:\n")


    # Segmentación de noticias negacionistas

    nlp = spacy.load("es_core_news_sm")
    noticias_negacionistas_segmentadas = []

    for noticia in noticias_negacionistas_no_duplicadas:
        doc = nlp(noticia)
        sentencias = list(doc.sents)

        for i in range(len(sentencias)):
            sentencias[i] = sentencias[i].text

        noticia = "\n\n".join(sentencias)
        print(noticia)
        print()

        try: 
            # TextTilling
            stopwords_spanish = set(stopwords.words('spanish'))
            tt = TextTilingTokenizer(stopwords=stopwords_spanish)
            segments = tt.tokenize(noticia)  
            
            for i in range(len(segments)):
                segment = segments[i]        
                noticias_negacionistas_segmentadas.append(segment)
                print(i, " ".join(segment.splitlines()))

            print("Texto segmentado en ",len(segments)," segmentos.\n")
            print()
        except:
            print("Demasiado corto para segmentarse.\n")
            print()
            noticias_negacionistas_segmentadas.append(noticia)    


with open('noticias_negacionistas_segmentadas.json', 'w', encoding='utf-8') as json_file:
    json.dump(noticias_negacionistas_segmentadas, json_file)


# Segunda parte (noticias no negacionistas)

with open('salida_no_negacionistas.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f

    print(f"Número de noticias no negacionistas: {len(noticias_no_negacionistas)}")

    # Eliminación de noticias no negacionista cuasi-duplicadas

    firmas = []
    valor_f = 128
    observaciones = []

    with tqdm(total=len(noticias_no_negacionistas)) as barra:
        for i in range(len(noticias_no_negacionistas)):
            texto = noticias_no_negacionistas[i]
            firma = Simhash(texto, f=valor_f)
            firmas.append((i,firma))    
            barra.update(1)

    indice = SimhashIndex(firmas, k=10, f=valor_f)

    with tqdm(total=len(noticias_no_negacionistas)) as barra:
        for i in range(len(noticias_no_negacionistas)):
            firma = firmas[i][1]
            duplicados = indice.get_near_dups(firma)
            observaciones.append(len(duplicados))
            barra.update(1)

    noticias_no_negacionistas_no_duplicadas = list()
    ignorar = list()

    with tqdm(total=len(noticias_no_negacionistas)) as barra:
        for i in range(len(noticias_no_negacionistas)):
            if i not in ignorar:
                texto = noticias_no_negacionistas[i]
                firma = firmas[i][1]
                duplicados = indice.get_near_dups(firma)

                if len(duplicados)==1:
                    noticias_no_negacionistas_no_duplicadas.append(texto)
                    ignorar.append(i)
                else:
                    random.shuffle(duplicados)
                    ident = int(duplicados[0])        
                    noticias_no_negacionistas_no_duplicadas.append(noticias_no_negacionistas[ident])

                    for ident in duplicados:
                        ignorar.append(int(ident))

            barra.update(1)
    
    print()
    print(f"Noticias no negacionistas tras eliminar las cuasi-duplicadas: {len(noticias_no_negacionistas_no_duplicadas)}")
    print()
    print()
    print("Segmentación de las noticias no negacionistas:\n")

    # Segmentación de noticias no negacionistas

    nlp = spacy.load("es_core_news_sm")
    noticias_no_negacionistas_segmentadas = []

    for noticia in noticias_no_negacionistas_no_duplicadas:
        doc = nlp(noticia)
        sentencias = list(doc.sents)

        for i in range(len(sentencias)):
            sentencias[i] = sentencias[i].text

        noticia = "\n\n".join(sentencias)

        try: 
            # TextTilling
            stopwords_spanish = set(stopwords.words('spanish'))
            tt = TextTilingTokenizer(stopwords=stopwords_spanish)
            segments = tt.tokenize(noticia)  
            
            for i in range(len(segments)):
                segment = segments[i]
                print(i, " ".join(segment.splitlines()))

            print("Texto segmentado en ",len(segments)," segmentos.\n")
            print()            
            noticias_no_negacionistas_segmentadas.append(segments) 
        except:
            print("Demasiado corto para segmentarse.\n")
            print()
            noticias_no_negacionistas_segmentadas.append(noticia)    


with open('noticias_no_negacionistas_segmentadas.json', 'w', encoding='utf-8') as json_file:
    json.dump(noticias_no_negacionistas_segmentadas, json_file)