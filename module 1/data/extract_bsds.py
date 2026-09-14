"""Build a tabular ML dataset from BSDS500: boundary vs non-boundary pixel classification.

Label policy (consensus labelling):
  positive  = >=2 human annotators drew a boundary within 1 px of the pixel
  negative  = NO annotator drew a boundary within 2 px
  ambiguous = everything else -> excluded (label noise control)
"""
import numpy as np, glob, os
from scipy.io import loadmat
from scipy import ndimage as ndi
from skimage import io, color

RNG = np.random.default_rng(0)
ROOT = "BSDS500-master/BSDS500/data"

FEATS = (["L","a","b"]
         + [f"gradmag_s{s}" for s in (1,2,4)]
         + [f"localstd_s{s}" for s in (1,2,4)]
         + [f"abslap_s{s}" for s in (1,2)]
         + [f"orient{d}_s2" for d in (0,45,90,135)]
         + ["st_eig1","st_eig2"])

def features_for(img):
    lab = color.rgb2lab(img)
    L = lab[...,0]
    F = [lab[...,0], lab[...,1], lab[...,2]]
    for s in (1,2,4):                                   # multi-scale gradient magnitude
        F.append(ndi.gaussian_gradient_magnitude(L, s))
    for s in (1,2,4):                                   # local contrast
        m  = ndi.gaussian_filter(L, s)
        m2 = ndi.gaussian_filter(L**2, s)
        F.append(np.sqrt(np.maximum(m2 - m**2, 0)))
    for s in (1,2):                                     # blob/ridge response
        F.append(np.abs(ndi.gaussian_laplace(L, s)))
    gy = ndi.gaussian_filter(L, 2, order=(1,0)); gx = ndi.gaussian_filter(L, 2, order=(0,1))
    for th in (0,45,90,135):                            # oriented derivative energy
        t = np.deg2rad(th)
        F.append(np.abs(np.cos(t)*gx + np.sin(t)*gy))
    # structure tensor eigenvalues (corner/edge energy)
    Jxx = ndi.gaussian_filter(gx*gx, 2); Jyy = ndi.gaussian_filter(gy*gy, 2); Jxy = ndi.gaussian_filter(gx*gy, 2)
    tr, dt = Jxx+Jyy, Jxx*Jyy-Jxy**2
    disc = np.sqrt(np.maximum((tr/2)**2 - dt, 0))
    F += [tr/2 + disc, tr/2 - disc]
    return np.stack(F, -1)

def labels_for(mat):
    gts = mat["groundTruth"][0]
    votes = np.zeros(gts[0]["Boundaries"][0,0].shape, int)
    anymark = np.zeros_like(votes, bool)
    for g in gts:
        B = g["Boundaries"][0,0].astype(bool)
        votes += ndi.binary_dilation(B, iterations=1)
        anymark |= ndi.binary_dilation(B, iterations=2)
    pos = votes >= 2
    neg = ~anymark
    return pos, neg

def sample_split(split, n_img, n_pos=160, n_neg=440, margin=8):
    Xs, ys, img_ids = [], [], []
    files = sorted(glob.glob(f"{ROOT}/images/{split}/*.jpg"))[:n_img]
    for f in files:
        iid = os.path.splitext(os.path.basename(f))[0]
        img = io.imread(f)
        mat = loadmat(f"{ROOT}/groundTruth/{split}/{iid}.mat")
        F = features_for(img)
        pos, neg = labels_for(mat)
        valid = np.zeros_like(pos); valid[margin:-margin, margin:-margin] = True  # avoid border artefacts
        for mask, lab, n in ((pos & valid, 1, n_pos), (neg & valid, 0, n_neg)):
            r, c = np.where(mask)
            if len(r) == 0: continue
            k = RNG.choice(len(r), min(n, len(r)), replace=False)
            Xs.append(F[r[k], c[k]]); ys.append(np.full(len(k), lab)); img_ids += [iid]*len(k)
    return np.vstack(Xs), np.hstack(ys), np.array(img_ids)

if __name__ == "__main__":
    X_tr, y_tr, id_tr = sample_split("train", 80)
    X_te, y_te, id_te = sample_split("test", 30)
    np.savez_compressed("bsds_pixels.npz", X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te,
                        id_tr=id_tr, id_te=id_te, features=np.array(FEATS))
    print("train:", X_tr.shape, "boundary rate:", round(float(y_tr.mean()),4))
    print("test :", X_te.shape, "boundary rate:", round(float(y_te.mean()),4))
    # CSV copy for the submission package
    import pandas as pd
    d = pd.DataFrame(np.vstack([X_tr, X_te]), columns=FEATS)
    d["BOUNDARY"] = np.hstack([y_tr, y_te]); d["split"] = ["train"]*len(y_tr)+["test"]*len(y_te)
    d["image_id"] = np.hstack([id_tr, id_te])
    d.to_csv("bsds_pixels.csv", index=False)
    print("csv rows:", len(d))
