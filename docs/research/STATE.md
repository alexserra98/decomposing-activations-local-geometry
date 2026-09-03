# Research State (updated 2026-09-03)

> **Kind:** Research state · **Status:** Current as of 2026-09-01 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

Apparently the model was already able to model the full tangent plane it just required to play a little bit with surgery_threshold and surgery_min_count. The latter in particular seems to be very important although this seems a bit fishy. I have also refined the alignment metric so that it computes also if the tange plane is included in the span of first q PCs of the gaussians. This second score is much more stable across the hyperparameter sweep. 
In order to chose the best model I have used modified BIC where I add a penalty for inactive components. The real reason is purely practical: the best model seems to keep more components alive, but it also intuitive in a sense given that if you manage to keep more components alive with the same NLL it means the model is correctly modelling the local geometry. This also requires to control the ratio size_dataset / K, but if you are confident that each gaussian is receiving enough points to model the geometry you should be ok.