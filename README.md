## decomposing-activations-local-geometry

This is the official repository for **“From Directions to Regions: Decomposing Activations in Language Models via Local Geometry”** (Or Shafran, Shaked Ronen, Omri Fahn, Shauli Ravfogel, Atticus Geiger, Mor Geva). 2026.

We’ve uploaded an **end-to-end tutorial** that walks through the core MFA workflow:

* **Training** an MFA on model activations
* **Interpreting** regions (centroids) and their **local directions of variation**
* **Visualizing** subspaces
* **Steering** using region-level structure and local subspaces

The repository uses a standard `src/` layout: reusable library code lives in
`src/dalg/`, runnable workflows are exposed as CLI entrypoints such as
`dalg-run-layer` and `dalg-interpret-mfa`, and generated artifacts live under
`outputs/`.

**Coming soon:** additional code to **recreate the paper experiments**, along with **released trained MFAs** for **Llama-3.1-8B** and **Gemma-2-2B**.

For any questions, feel free to reach out!
