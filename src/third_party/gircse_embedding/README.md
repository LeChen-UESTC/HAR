# Vendored GIRCSE Embedding Code

Source: https://github.com/Roytsai27/GIRCSE/tree/main/embedding

Copied from commit:

```text
20676c15294e161bcfd5d5be97e75498e54fdb8f
```

License: MIT, copied in `LICENSE`.

The project integration uses the same core algorithm as
`embedding/trainer.py`:

- iterative soft-token extension
- `softmax(logits / logit_temperature) @ embedding_weight`
- generated-token hidden-state collection
- `last` or `generate_mean` pooling

Copied files are `base.py`, `trainer.py`, and `model.py`. Local adapters live
in `src/models/gircse_adapter.py` so skeleton tokens can enter the same
generation mechanism through `inputs_embeds`.
