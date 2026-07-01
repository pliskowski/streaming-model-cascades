# Benchmark datasets

Six CSV files used by the paper experiments. Each row is one inference query scored offline by a proxy model and an oracle model. Cascade algorithms only make routing decisions; they do not call any LLM at runtime.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `proxy_pred_true_prob` | float in `[0, 1]` | Proxy estimate of `P(label = 1)`. |
| `oracle_pred` | `true` / `false` | Oracle label (ground truth for evaluation). |

No raw text from any source corpus is included.

## Files

| File | Rows | Upstream source |
|---|---|---|
| `MMLU_5K.csv` | 5,000 | [MMLU](https://arxiv.org/abs/2009.03300) (binary correctness) |
| `BOOLQ_P0_12697.csv` | 12,697 | [BoolQ](https://arxiv.org/abs/1905.10044) |
| `IMDB_50000.csv` | 50,000 | [IMDB](https://ai.stanford.edu/~ang/papers/acl11-WordVectors.pdf) |
| `ARXIV_56180.csv` | 56,181 | [arXiv subjects](https://arxiv.org/abs/1704.01212) |
| `SST2_68221.csv` | 68,221 | [SST-2](https://nlp.stanford.edu/~sidaw/home/projects:npsentiment) |
| `NYT_500K.csv` | 250,000 | Scores derived from the [NYT Annotated Corpus](https://catalog.ldc.upenn.edu/LDC2008T19) (Sandhaus, 2008); source text not redistributed |

Proxy scores were produced with Llama 3.1-8B; oracle labels with Llama 3.3-70B.

## License

Derived score files in this directory are released under the MIT license (see repository [LICENSE](../LICENSE)). Upstream datasets remain under their original terms.
