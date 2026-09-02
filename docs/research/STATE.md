# Research State (updated 2026-09-01)

> **Kind:** Research state · **Status:** Current as of 2026-09-01 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

Moving to more complicated manifolds such as the torus some problems started to emerge.
In the past log I had observed that less populated gaussian tended to have a poorer recostruction of tangent space, but I have later realized that this is partially false. Now I have a better metric to measure alignment between PCs and tangent plane of the manifold and I was able to study better the hddc models sweeped over surgery_threshold. Apparently with surgery_threshold=0.1 I was able to obtain a very good alignment for helix and circumference and modest for torus. By visual inspection you I could see that in all three manifold the first PC is succesfully capturing a direction of the tangent plane and notably this improve as I increase surgery_threshold. When surgery_threshold >= 1 in particular each gaussian is forced to have d = 1 (look the docs to understand the notation) and at least from visual inspection the torus is tiled really well, that is the single PC is perfectly aligned with one direction of the tangent plane. The issue though is that I would like the gaussians to model the full tangent plane and right now my model is not able to that. 

