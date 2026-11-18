# ML examples

Small machine-learning programs written in Klong: nearest neighbours, cosine similarity, a transformer and an LSTM. This page is for someone who wants to run them.

## Requirements

* `knn.kg`, `cosine_similarity.kg` and `transformer.kg` need NumPy, which KlongPy installs.
* `lstm/lstm.kg` also needs PyTorch: `pip install torch`.

## Run

From this directory:

```bash
kgpy knn.kg
kgpy cosine_similarity.kg
kgpy transformer.kg
```

The LSTM loads its Python files by relative path, so run it from its own directory:

```bash
cd lstm
kgpy lstm.kg
```

`faiss/` has a nearest-neighbour search over a flat index; see its own README.
