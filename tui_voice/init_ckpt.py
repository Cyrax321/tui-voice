"""Apply a vocab plan to an upstream checkpoint: row `slot` of every symbol-indexed table
becomes the row of init_from[slot]. Slots that kept their symbol are unchanged.
Works on torch tensors or numpy arrays. After this the checkpoint has the SAME shapes as
upstream, so tiny-voca's train code can load it unchanged."""

def remap_embeddings(state, init_from, key_hints=("emb", "symbol", "phon", "token"), n=None):
    n = n or len(init_from)
    changed = []
    out = dict(state)
    for k, v in state.items():
        shp = getattr(v, "shape", None)
        if shp is None or len(shp) < 1 or shp[0] != n:
            continue
        if not any(h in k.lower() for h in key_hints) or len(shp) not in (2,):
            continue                       # only 2-D tables whose name looks like a symbol embedding
        w = v.clone() if hasattr(v, "clone") else v.copy()
        for slot, src in enumerate(init_from):
            if slot != src:
                w[slot] = v[src]
        out[k] = w
        changed.append(k)
    return out, changed
