# Research State (updated 2026-09-05)

> **Kind:** Research state · **Status:** Current as of 2026-09-05 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

Ok apparently I have found a solution. I created 5 copies dataset with 10 manifolds varying noise ratio in this window [10000, 1000, 100, 10]. The lower the noisier. For 10000,1000,100 both kmeans and MFA are able to recover tangent space and tile the manifold. For 10 instead kmeans fails on very simple manifold, such a line, or the sphere, while MFA works decently. So I have a reason to justify MFA: it works better with noisy dataset. The best would be to have an other dataset where kmeas fails and MFA works.