# Research State (updated 2026-11-11)

> **Kind:** Research state · **Status:** Current as of 2026-11-11 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

We have updated the PC1 (at the current moment is still on the origin/codex/PC1-density-alignment branch) correcting some conceptual flaws and we managed to prove that hddc beats kmeans in all the benchmark.
I have implemented the label alignment metric and after some corrections also for that we get hddc winning over kmeans in most of the times.The problem with this new metric is that is not trivial to extend it to other label when there is no clear ordering and also it depends on the granularity of the label. For coarse granularity hddc perform very well, for fine grained labeling kmeans is slightly better or comparable with hddc, although both perform poorly. I think it's an interesting benchmark because it allows us to qualify the semantic meaning of these gaussian PC1. Also if we project acts on 3 PCs the gap between this 2 models shrinks while full dimensional acts favour hddc. I take this as a sign that hddc help taming the curse of dimensionality.


