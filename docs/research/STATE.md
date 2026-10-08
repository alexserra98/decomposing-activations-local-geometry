# Research State (updated 2026-10-08)

> **Kind:** Research state · **Status:** Current as of 2026-09-29 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

I have fixed a majory issue with the way surgery was computed. Before when no gap where found for some reason I was putting q_k=1 which made no sense. Now when no gap is found q_k is set to q_max and then adjusted further by comparing with the noise b_k. Directions with eigenvalue $< b_k$ where removed from a_ij and the noise was recomputed.

I have explored how the model behave on torus12d running different experiments. The problem is the following: we can get a good adjusted tantent alignment but we cannot recover the ID. I tried:
- increasing K and number of points doesn't improve much the results at least for the scale I have currently tried
- I have tried to increase the number of points but still can get the ID right
- I tried reducing the dimensionality of the torus and for $dim<6$ I can recoover ID.
An important thing I have observed is that using a dataset with only a torus6d and 5M of points I was able to reduce NLL but getting worse performance than with a dataset of 500k. Looking further I have discovered that the 5M was reaching an high adjusted tangetn alignment in the first 2 iteration and then the training improved NLL but degraded alignment. This is yet an other evidence that NLL is not a good metric for our problem. 

I tried plotting the spectrum for different manifolds and I have observed a couple interesting things:
- For smaller id manifolds the spectrum has two jump, one in correspondece of ID and the other of manifold dim.
- For the torus there was no jump for the ID in the 30k dataset.
- In 5M dataset the jump was more visible.