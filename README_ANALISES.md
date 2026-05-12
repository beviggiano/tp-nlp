# README de Analises e Modelos

Este repositorio entrega uma base experimental para investigar a pergunta do
projeto:

> Metricas baseadas em propensity scoring sao robustas contra um baseline
> ingenuo baseado apenas na frequencia global dos rotulos em cenarios de
> Extreme Multi-Label Text Classification (XMTC)?

A ideia central e comparar um modelo que nao le texto nenhum contra um modelo
que usa recuperacao textual. Se o baseline por frequencia tiver desempenho
forte em metricas tradicionais, mas cair em metricas sensiveis a rotulos raros,
isso indica que as metricas com propensity scoring estao capturando melhor o
problema de desbalanceamento.

## O que o codigo entrega

O codigo entrega quatro blocos principais:

- Um baseline ingenuo chamado `GlobalFrequencyRanker`.
- Um modelo textual de recuperacao chamado `InstanceKNNRetrievalRanker`.
- Um motor de metricas com `Precision@k`, `nDCG@k`, `PS-Precision@k` e
  `PS-nDCG@k`.
- Um modulo de analise para comparar desempenho geral e desempenho separado em
  rotulos head, middle e tail.

A organizacao foi feita para que qualquer modelo novo possa substituir os
rankers atuais, desde que produza uma matriz de predicoes top-k com ids de
rotulos.

## Estrutura do repositorio

```text
src/xmtc_robustness/
  models/
    global_frequency.py   # baseline que ranqueia rotulos por frequencia global
    retrieval.py          # modelo de recuperacao textual via k-NN
    base.py               # protocolo comum para rankers top-k
  metrics.py              # metricas tradicionais e propensity-scored
  analysis.py             # comparacao geral e por head/tail
  data.py                 # carregamento de datasets locais

scripts/
  run_local_experiment.py # runner para testar uma base local

data/
  README.md               # formato esperado da base de dados

tests/
  test_*.py               # testes unitarios leves
```

## Formato dos dados

O repositorio espera que a base seja convertida para este formato:

```text
data/nome_da_base/
  train_texts.txt
  test_texts.txt
  y_train.npz
  y_test.npz
  labels.txt        # opcional
```

`train_texts.txt` e `test_texts.txt` guardam um documento por linha.
`y_train.npz` e `y_test.npz` sao matrizes esparsas SciPy CSR com shape:

```text
y_train: n_train x n_labels
y_test:  n_test  x n_labels
```

Cada coluna representa um rotulo. Cada linha representa uma instancia. Um valor
na posicao `(i, j)` indica que a instancia `i` possui o rotulo `j`.

## Modelo 1: GlobalFrequencyRanker

Arquivo: `src/xmtc_robustness/models/global_frequency.py`

Este e o baseline ingenuo do projeto. Ele ignora completamente o conteudo
textual dos documentos.

### Como ele e treinado

Durante o `fit(y_train)`, o modelo soma cada coluna de `y_train`:

```text
freq(label_j) = soma_i y_train[i, j]
```

Isso produz um vetor de frequencias globais dos rotulos. Em seguida, os rotulos
sao ordenados em ordem decrescente de frequencia. Em caso de empate, o menor id
de rotulo vem primeiro, garantindo resultado deterministico.

### Como ele prediz

Para qualquer documento de teste, o modelo retorna exatamente os mesmos top-k
rotulos mais frequentes do treino.

Exemplo conceitual:

```text
ranking global = [label_7, label_2, label_10, label_1]

predicao para doc A = [label_7, label_2, label_10]
predicao para doc B = [label_7, label_2, label_10]
predicao para doc C = [label_7, label_2, label_10]
```

### Por que ele importa na pesquisa

Ele mede o quanto as metricas podem ser enganadas por popularidade. Se uma
metrica tradicional der resultado alto para esse baseline, isso sugere que o
protocolo de avaliacao pode estar favorecendo rotulos frequentes em vez de
medir compreensao textual real.

### Escalabilidade

O baseline usa operacoes vetorizadas:

- Soma de colunas em matriz esparsa.
- Ordenacao unica dos rotulos.
- `np.broadcast_to` para repetir o mesmo top-k para milhares de instancias.

Ele nao faz loop pesado por documento de teste.

## Modelo 2: InstanceKNNRetrievalRanker

Arquivo: `src/xmtc_robustness/models/retrieval.py`

Este e o modelo textual principal. Ele trata XMTC como um problema de
recuperacao: dado um texto de teste, busca documentos de treino semanticamente
parecidos e agrega os rotulos desses vizinhos.

### Etapa 1: representacao textual

O modelo aceita duas familias de representacao:

- `tfidf`: usa `TfidfVectorizer` do scikit-learn.
- `sentence-transformer`: usa embeddings densos de modelos como
  `sentence-transformers/all-MiniLM-L6-v2`.

Com TF-IDF, os vetores podem ser esparsos. Com Sentence-Transformers, os vetores
sao densos e normalizados.

### Etapa 2: indexacao de vizinhos

Depois de transformar os textos de treino em vetores, o modelo constroi um
indice de busca por vizinhos proximos. Os backends suportados sao:

- `sklearn`: busca exata com `NearestNeighbors` e distancia de cosseno.
- `faiss`: busca vetorial densa com produto interno.
- `hnswlib`: busca aproximada com grafo HNSW.
- `auto`: escolhe FAISS/HNSW quando possivel e usa sklearn como fallback.

Para bases grandes como Amazon-670K, a rota ideal e usar embeddings densos com
FAISS ou HNSWlib.

### Etapa 3: predicao de rotulos

Para cada texto de teste:

1. O texto e convertido para vetor.
2. O indice recupera os `n_neighbors` documentos de treino mais proximos.
3. Os rotulos desses vizinhos sao somados com peso proporcional a similaridade.
4. Os top-k rotulos com maior score agregado sao retornados.

Formalmente, para uma consulta `q`:

```text
score(label_l | q) = soma_{v em vizinhos(q)} sim(q, v) * y_train[v, l]
```

Se nenhum rotulo recebe score positivo, o modelo usa o ranking de frequencia
global como fallback.

### Por que ele importa na pesquisa

Ele e o contraponto textual do baseline. A expectativa e que ele recupere mais
rotulos relevantes, especialmente quando a similaridade textual aponta para
rotulos menos frequentes que o baseline nao priorizaria.

## Motor de metricas

Arquivo: `src/xmtc_robustness/metrics.py`

O motor recebe:

```text
y_true: matriz real de rotulos
y_pred: matriz n_samples x k com ids de rotulos preditos
```

Todas as metricas operam sobre a mesma matriz de predicao. Isso garante que a
comparacao entre metricas seja justa.

## Precision@k

`Precision@k` mede a fracao dos k rotulos preditos que estao corretos:

```text
Precision@k = acertos_no_top_k / k
```

Ela e simples e interpretavel, mas pode favorecer modelos que sempre predizem
rotulos populares, porque esses rotulos aparecem em muitas instancias.

## nDCG@k

`nDCG@k` avalia ranking: acertos nas primeiras posicoes valem mais do que
acertos nas ultimas posicoes.

O ganho descontado e:

```text
DCG@k = soma_i rel_i / log2(i + 1)
```

Depois o valor e normalizado pelo ranking ideal:

```text
nDCG@k = DCG@k / IDCG@k
```

No codigo, a relevancia e binaria: um rotulo predito esta correto ou nao.

## Propensity scoring

As metricas com propensity scoring usam pesos maiores para rotulos raros. O
codigo segue a formulacao de Jain et al.:

```text
p_l = 1 / (1 + C * exp(-a * log(n_l + b)))
C = (log(N) - 1) * (b + 1)^a
```

Onde:

- `p_l` e a propensity do rotulo `l`.
- `n_l` e a frequencia do rotulo no treino.
- `N` e o numero de instancias de treino.
- `a` e `b` controlam a curva de penalizacao.

O codigo usa o inverso da propensity:

```text
inverse_propensity_l = 1 / p_l
```

Assim, um acerto em rotulo raro vale mais do que um acerto em rotulo frequente.

Presets implementados:

```text
default / wiki10-31k: a=0.55, b=1.5
wikilshtc:            a=0.5,  b=0.4
amazon-670k:          a=0.6,  b=2.6
amazoncat-13k:        a=0.6,  b=2.6
```

## PS-Precision@k

`PS-Precision@k` substitui o ganho binario por ganho ponderado:

```text
PS-Precision@k = soma_{rotulo correto no top-k} inverse_propensity_l / k
```

Essa metrica responde: o modelo acertou apenas rotulos populares ou tambem
acertou rotulos raros?

## PS-nDCG@k

`PS-nDCG@k` combina duas ideias:

- Ranking: posicoes mais altas importam mais.
- Propensity: rotulos raros corretos valem mais.

O DCG ponderado e:

```text
PS-DCG@k = soma_i rel_i * inverse_propensity(label_i) / log2(i + 1)
```

O denominador ideal tambem e ponderado: ele coloca os rotulos verdadeiros de
maior inverse propensity nas melhores posicoes.

## Analise head, middle e tail

Arquivo: `src/xmtc_robustness/analysis.py`

O codigo separa os rotulos por frequencia no treino:

- `head`: rotulos mais frequentes.
- `middle`: rotulos intermediarios.
- `tail`: rotulos mais raros.

Por padrao:

```text
head = top 20% dos rotulos com frequencia positiva
tail = bottom 20% dos rotulos com frequencia positiva
middle = restante
```

A funcao `evaluate_by_frequency_partition` avalia a mesma matriz de predicoes
em cada subconjunto. Isso permite observar, por exemplo:

- O baseline ganha em `head/precision@k`?
- O baseline perde em `tail/ps_ndcg@k`?
- O modelo de recuperacao melhora na cauda?
- A metrica tradicional mascara uma diferenca que aparece na metrica PS?

## Comparacao entre modelos

A funcao `compare_predictions` recebe um dicionario de predicoes:

```python
{
    "global_frequency": baseline_pred,
    "retrieval": retrieval_pred,
}
```

Ela retorna uma tabela em formato de dicionario com metricas gerais e, se
solicitado, metricas por particao de frequencia.

Exemplo de saida esperada:

```text
[global_frequency]
precision@1: 0.420000
ndcg@5: 0.310000
ps_precision@5: 0.080000
tail/ps_ndcg@5: 0.010000

[retrieval]
precision@1: 0.390000
ndcg@5: 0.330000
ps_precision@5: 0.160000
tail/ps_ndcg@5: 0.090000
```

Um padrao como esse indicaria que o baseline pode parecer competitivo em
metricas tradicionais, mas perde quando a avaliacao da mais importancia a
rotulos raros.

## Analises que o repositorio suporta

### 1. Robustez contra baseline ingenuo

Compara `GlobalFrequencyRanker` e `InstanceKNNRetrievalRanker` nas mesmas
divisoes de treino/teste. Essa e a analise principal do projeto.

Pergunta respondida:

```text
As metricas com propensity scoring reduzem a vantagem artificial do baseline
baseado em frequencia?
```

### 2. Sensibilidade a rotulos raros

Usa as particoes `head`, `middle` e `tail` para verificar onde cada modelo esta
acertando.

Perguntas respondidas:

```text
O modelo ganha porque acerta rotulos frequentes?
O ganho em PS-nDCG vem de rotulos da cauda?
O baseline praticamente desaparece na particao tail?
```

### 3. Comparacao entre metricas tradicionais e PS

Executa as quatro metricas sobre as mesmas predicoes.

Pergunta respondida:

```text
Precision@k e nDCG@k contam uma historia diferente de PS-Precision@k e
PS-nDCG@k?
```

### 4. Analise de desbalanceamento

O codigo esta preparado para avaliar diferentes versoes de uma mesma base, por
exemplo:

```text
data/amazon_original/
data/amazon_head_amplificado/
data/amazon_tail_removido/
```

Cada pasta pode ser rodada pelo mesmo script. A comparacao dos resultados mostra
como as metricas mudam quando o desbalanceamento e intensificado.

## Como rodar uma analise local

Instale o pacote:

```bash
pip install -e .
```

Rode apenas o baseline:

```bash
python scripts/run_local_experiment.py --data data/minha_base --models baseline --ks 1 3 5 --dataset-preset amazon-670k
```

Rode baseline e recuperacao TF-IDF:

```bash
python scripts/run_local_experiment.py --data data/minha_base --models baseline retrieval --ks 1 3 5 --representation tfidf --retrieval-backend sklearn --dataset-preset amazon-670k
```

Com embeddings densos e FAISS:

```bash
pip install -e ".[faiss,embeddings]"

python scripts/run_local_experiment.py --data data/minha_base --models baseline retrieval --ks 1 3 5 --representation sentence-transformer --retrieval-backend faiss --dataset-preset amazon-670k
```

## Como interpretar os resultados

Um resultado preocupante para a robustez seria:

```text
global_frequency >= retrieval em PS-Precision@k e PS-nDCG@k
```

Isso indicaria que mesmo as metricas com propensity scoring ainda podem ser
vulneraveis ao ranking por frequencia.

Um resultado favoravel as metricas PS seria:

```text
global_frequency competitivo em Precision@k
global_frequency fraco em PS-Precision@k e tail/PS-nDCG@k
retrieval melhor em metricas PS e na cauda
```

Isso indicaria que as metricas PS estao penalizando a dependencia excessiva de
rotulos frequentes.

## Limitacoes atuais

O repositorio implementa o nucleo experimental, mas ainda nao inclui:

- Download automatico de WikiLSHTC ou Amazon-670K.
- Conversores especificos para cada formato bruto de dataset.
- Treinamento de dual-encoder supervisionado.
- Relatorios automaticos em CSV/LaTeX.
- Graficos de distribuicao de frequencia e curvas por grau de desbalanceamento.

Esses pontos podem ser adicionados como proximas etapas sem alterar o desenho
central da pipeline.

## Resumo cientifico

O desenho experimental e deliberadamente simples:

```text
baseline sem texto
vs.
modelo com texto
avaliados por metricas tradicionais
vs.
metricas com propensity scoring
em rotulos head/middle/tail
```

Essa estrutura isola a pergunta de pesquisa. Se uma metrica e robusta, ela deve
mostrar que um modelo que apenas repete rotulos frequentes nao resolve XMTC de
forma confiavel, especialmente quando a avaliacao enfatiza os rotulos raros.
