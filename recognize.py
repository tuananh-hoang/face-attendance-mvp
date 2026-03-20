import numpy as np

def recognize(face_emb, db_embs, db_names, thresh=0.6):
    sims = db_embs @ face_emb
    idx = np.argmax(sims)
    score = sims[idx]

    if score >= thresh:
        return db_names[idx], float(score)
    return "UNKNOWN", float(score)
