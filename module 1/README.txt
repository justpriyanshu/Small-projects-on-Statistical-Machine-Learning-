ED408 - Statistical Machine Learning II Lab (Module I)
Dataset: Berkeley Segmentation Dataset (BSDS500)
Task: boundary vs non-boundary pixel classification (17 local features,
      consensus human labels, train/test pixels from disjoint images)

Contents
--------
ED408_Submission.pdf ................. compiled lab file in the required format
latex/ED408_Submission.tex ........... LaTeX source (compile: pdflatex x2, needs latex/figs/)
latex/figs/ .......................... figures used by the LaTeX file
notebooks/Lab01..Lab07 ............... 7 standalone, fully executed notebooks (one per experiment)
notebooks/ED408_Module1_All_Labs_Combined.ipynb ... all 7 labs in one notebook
data/bsds_pixels.npz ................. extracted pixel features + labels (48k train / 18k test)
data/extract_bsds.py ................. feature/label extraction pipeline

Re-running
----------
Each lab notebook is self-contained; keep data/bsds_pixels.npz next to the
notebook (or fix the path in the first code cell). Kernel: Python 3 with
numpy, pandas, matplotlib, seaborn, scikit-learn.

To rebuild bsds_pixels.npz from scratch, download BSDS500 (e.g. the BIDS/BSDS500
GitHub mirror), place extract_bsds.py next to BSDS500-master/, and run it
(needs scipy + scikit-image). Raw images are not included here for size.
