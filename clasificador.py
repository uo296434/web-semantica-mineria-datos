import json
import random
import fasttext
import sys

#0. Energía
#1. Agua 
#15. Emisiones
#18. Incendios forestales
#22. Covid-19

# Primera colección (noticias negacionistas)

with open('noticias_segmentadas.json', 'r', encoding='utf-8') as json_file:
    noticias_negacionistas_segmentadas = json.load(json_file)

noticias_negacionistas_segmentadas = [line.replace('\n', '') for line in noticias_negacionistas_segmentadas]

with open('clustered_docs.json', 'r', encoding='utf-8') as json_file:
    clustered_docs = json.load(json_file)

clustered_docs = {int(k): v for k, v in clustered_docs.items()}
selected_clusters = [0, 1, 15, 18, 22]
selected_clustered_docs = {clave: valores for clave, valores in clustered_docs.items() if clave in selected_clusters}
etiquetas = {0: "__label__energia", 1: "__label__agua", 15: "__label__emisiones", 18: "__label__incendiosforestales", 22: "__label__covid19"}

with open('datos_etiqueta_segmento.txt', 'w', encoding='utf-8') as f:
    for clave, valores in selected_clustered_docs.items():
        etiqueta = etiquetas[clave]
        for valor in valores:
            segmento = noticias_negacionistas_segmentadas[valor]
            f.write(f'{etiqueta} {segmento}\n')

with open('datos_etiqueta_segmento.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

random.shuffle(lines)
split_index = int(0.8 * len(lines))
train_lines = lines[:split_index]
test_lines = lines[split_index:]

with open('train.txt', 'w', encoding='utf-8') as f:
    f.writelines(train_lines)

with open('test.txt', 'w', encoding='utf-8') as f:
    f.writelines(test_lines)

with open('resultados_clasificador.txt', 'w', encoding='utf-8') as f:
    sys.stdout = f

    clasificador = fasttext.train_supervised('train.txt')
    result = clasificador.test('test.txt')

    print(f"Resultados clasificador: {result}")


# Segunda colección (noticias no negacionistas)

with open('noticias_no_negacionistas_segmentadas.json', 'r', encoding='utf-8') as json_file:
    noticias_no_negacionistas_segmentadas = json.load(json_file)

noticias_con_temas = []

for noticia in noticias_no_negacionistas_segmentadas:
    segmentos = noticia
    temas_por_segmento = []
    for segmento in segmentos:
        segmento = segmento.replace('\n', '')
        tema = clasificador.predict(segmento)
        temas_por_segmento.append(tema[0][0].replace('__label__', ''))
    noticias_con_temas.append({'noticia': noticia, 'temas': temas_por_segmento})

for noticia_con_temas in noticias_con_temas:
    segmentos_por_tema = {}
    total_segmentos = len(noticia_con_temas['temas'])
    for tema in noticia_con_temas['temas']:
        if tema in segmentos_por_tema:
            segmentos_por_tema[tema] += 1
        else:
            segmentos_por_tema[tema] = 1
    etiquetas = {tema: segmentos_por_tema[tema] / total_segmentos for tema in segmentos_por_tema}
    noticia_con_temas['etiquetas'] = etiquetas
    del noticia_con_temas['temas']

with open('noticias_con_temas.ndjson', 'w', encoding='utf-8') as ndjson_file:
    for noticia_con_temas in noticias_con_temas:
        json.dump(noticia_con_temas, ndjson_file)
        ndjson_file.write('\n')