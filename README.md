# Small Projects on Statistical Machine Learning

Lab work for **ED408 – Statistical Machine Learning II Lab**. Each experiment is a standalone, fully executed Jupyter notebook, and every module has a lab file (PDF) with the code, outputs and analysis.

- **Module 1:** classic ML techniques on one real problem, detecting object boundaries in BSDS500 images.
- **Module 2:** metaheuristic optimisation (SA, GA, PSO, ACO), regularised regression and Kernel PCA.

## Repository layout

```
.
├── module 1/          Experiments 1–7 (notebooks only)
├── module 2/          Experiments 8–13 + class exercises (notebooks)
├── data/              Dataset for module 1 and the script that builds it
└── docs/
    ├── module 1/      Lab file (PDF) + figures, module 1 README
    └── module 2/      Lab file (PDF) + figures, SA/GA worked solutions (PDF, DOCX)
```

## Module 1: Overview of Machine Learning Techniques

All seven experiments use the same task. Given 17 local image features of a pixel (colour, multi-scale gradients, Laplacian, oriented energy, structure tensor), predict whether human annotators marked an **object boundary** there. Data comes from the Berkeley Segmentation Dataset (BSDS500): 48,000 training and 18,000 test pixels, taken from disjoint images.

| Exp. | Notebook | What it covers |
|---|---|---|
| 1 | [k-Nearest Neighbours from scratch](module%201/Lab01_kNN_from_Scratch.ipynb) | Vectorised k-NN, effect of feature scaling, choosing *k*, distance-weighted voting |
| 2 | [Decision Tree](module%201/Lab02_Decision_Tree.ipynb) | Cost-complexity pruning, tree visualisation, feature importances |
| 3 | [Naive Bayes](module%201/Lab03_Naive_Bayes.ipynb) | Gaussian NB, log transform for heavy tails, NB vs QDA (the cost of the independence assumption) |
| 4 | [Logistic Regression from scratch](module%201/Lab04_Logistic_Regression_Scratch.ipynb) | Gradient descent, L2 penalty, class weighting, check against scikit-learn |
| 5 | [PCA and t-SNE](module%201/Lab05_PCA_tSNE.ipynb) | Explained variance, 2-D visualisation of the feature space |
| 6 | [Outlier Detection](module%201/Lab06_Outlier_Detection.ipynb) | Z-score vs robust z vs IQR, masking effect, winsorisation judged downstream |
| 7 | [Logistic Regression vs SVM](module%201/Lab07_LogReg_vs_SVM.ipynb) | Accuracy, precision, recall, F1, ROC-AUC, confusion matrices |
| — | [All labs combined](module%201/ED408_Module1_All_Labs_Combined.ipynb) | Experiments 1–7 in a single notebook |

## Module 2: Optimisation, Regularisation and Kernel Methods

| Exp. | Notebook | What it covers |
|---|---|---|
| 8 | [Simulated Annealing](module%202/Lab08_Simulated_Annealing.ipynb) | Minimise f(x) = x² + 10 sin(x); escaping a local minimum; effect of the cooling rate; comparison with hill climbing |
| 9 | [Genetic Algorithm for TSP](module%202/Lab09_Genetic_Algorithm_TSP.ipynb) | 25-city tour with order crossover, inversion mutation and elitism; comparison with nearest-neighbour |
| 10 | [PSO on Rosenbrock](module%202/Lab10_PSO_Rosenbrock.ipynb) | Swarm snapshots on the contour plot, decreasing inertia, scaling to 5-D and 10-D |
| 11 | [L1, L2 and Elastic Net](module%202/Lab11_L1_L2_ElasticNet_Regularization.ipynb) | Overfitting with n=60, p=40; sparsity; regularisation paths; diabetes dataset |
| 12 | [Kernel PCA](module%202/Lab12_Kernel_PCA.ipynb) | RBF Kernel PCA from scratch (checked against scikit-learn) on circles and moons; effect of γ |
| 13 | [Ant Colony Optimisation](module%202/Lab13_Ant_Colony_Optimization.ipynb) | Shortest path on a road network; Ant System vs MAX–MIN Ant System; checked against Dijkstra |

Class exercises:

- [PSO.ipynb](module%202/PSO.ipynb): PSO hand-worked class problem (maximise −x² + 5x + 20 with 9 particles).
- [SA_GA_Solutions.ipynb](module%202/SA_GA_Solutions.ipynb): worked numerical problems for Simulated Annealing and binary-encoded Genetic Algorithms. A write-up is in [docs/module 2](docs/module%202/SA_GA_Solutions.pdf).

## Lab files (docs)

| Module | Lab file | Figures |
|---|---|---|
| 1 | [sml_lab_module1.pdf](docs/module%201/sml_lab_module1.pdf) | [docs/module 1/figs](docs/module%201/figs) |
| 2 | [sml_lab_module2.pdf](docs/module%202/sml_lab_module2.pdf) | [docs/module 2/figs](docs/module%202/figs) |

Each lab file has the same sections for every experiment: Aim, Problem Statement, Algorithm, important program sections, output, a parameter-modification table and a conclusion. The `figs/` folders hold the figures used in each lab file.

## Running the notebooks

Requirements: Python 3 with

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

- **Module 1** notebooks load `../data/bsds_pixels.npz`, so open them from inside the `module 1/` folder.
- **Module 2** notebooks generate their own data (or use scikit-learn's built-in datasets) and need no files.
- Everything is seeded, so re-running a notebook reproduces the numbers in the lab files.

### Rebuilding the Module 1 dataset (optional)

[`data/bsds_pixels.npz`](data/bsds_pixels.npz) is already included. To rebuild it from the raw images:

1. Download BSDS500 (e.g. the BIDS/BSDS500 GitHub mirror).
2. Place [`extract_bsds.py`](data/extract_bsds.py) next to the `BSDS500-master/` folder.
3. Run it (needs `scipy` and `scikit-image`).

More details are in the [Module 1 README](docs/module%201/README.txt).
