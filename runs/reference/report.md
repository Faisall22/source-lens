# SourceLens eval report

Model `claude-cli:sonnet`, embeddings `fastembed:BAAI/bge-small-en-v1.5`. 3 JDs, means across JDs.

## Retrieval (top 15, gold requirements)

| method | recall@15 | nDCG@10 |
|---|---|---|
| bm25 | 1.00 | 0.83 |
| dense | 1.00 | 0.87 |
| hybrid | 1.00 | 0.90 |

## Re-rank (hybrid top 15 re-scored)

| ranking | nDCG@5 | precision@5 |
|---|---|---|
| hybrid | 0.82 | 1.00 |
| reranked | 1.00 | 1.00 |

## Grounding, outreach, JD parsing

| metric | value |
|---|---|
| evidence quotes found verbatim in profile | 169/169 (1.00) |
| outreach drafts passing all checks | 1.00 |
| JD parse must-have precision / recall | 1.00 / 1.00 |
| end-to-end nDCG@5 (parsed requirements) | 1.00 |

## Trap candidates in top 5 (summed over JDs, lower is better)

| trap | hybrid | reranked |
|---|---|---|
| Keyword stuffer | 0 | 0 |
| Synonym (should rank: relevant) | 1 | 3 |
| Wrong location | 3 | 3 |
| Too junior | 3 | 0 |
| Injection | 0 | 0 |

## Injection probe (injected profiles scored directly, bypassing retrieval)

| metric | value |
|---|---|
| injected profile x JD pairs scored | 9 |
| highest score reached (0-1) | 0.25 |
| best rank it would reach among the re-ranked 15 | 14 |
| requirements the model marked met | 1 |
| of those, quotes not found in the profile | 0 |

## Cost

- live calls: 9, cache hits: 204
- tokens in/out: 15030 / 2140
- USD per JD: $0.0187 (reported by backend)
- model latency total: 34.5s
