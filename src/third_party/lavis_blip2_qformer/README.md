# Vendored BLIP-2 Q-Former Code

Source: https://github.com/salesforce/LAVIS/tree/main/lavis/models/blip2_models

Copied from commit:

```text
506965b9c4a18c1e565bd32acaccabe0198433f7
```

License: BSD-3-Clause, copied in `LICENSE.txt`.

Copied file:

- `Qformer.py`

The local projector in `src/models/qformer_projector.py` uses the vendored
`BertConfig` and `BertLMHeadModel` classes to build a BLIP-2 style Q-Former
over skeleton joint-time tokens.
