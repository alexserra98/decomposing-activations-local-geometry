# Research State (updated 2026-10-08)

> **Kind:** Research state · **Status:** Current as of 2026-10-27 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

After playing around with multidimensional toruses I again observed that in the noiseless case that kmeans is doing much better than EM. Now it's clear that the only scenario where EM has an edge is in the noisy case. 
Since the only reason for keeping this model is hanging on how much the real world is noisy I though it was about time to start try with real activations. I have used the dataset of https://github.com/goodfire-ai/sae-manifold which collect a bunch of features manifold already known in the literature.
For now I restricted my analysis to temperature and years dataset. The first issue was that I needed to extend the dataset while keeping the manifold intact. I experimented a bit with the prompts, the structure was always [context]+label. While the in the original dataset the context is basically absent even in second half of the network you can see the manifold associated with the label. Since to extend the data I had to use different contexts I hava observed that in my prompts the manifold was visibile only in the first layer while in the last layer the activation seems to cluster by context although within the clusters there was some structure releated to the label. 
The clustering with hddc and kmeans provided some what different results although it's hard to evaluate. Both methods tile the manifold correctly, the hddc methods seems to have PC more aligned with the manifold but I can say much more than that. Some immediate problem I need to solve:
- [] How to evaluate our model on real dataset? We don't know neither the real ID neither the tangent plane
- [] How to scale EM?
- [] How to merge the tiles?

Now some more general considerations:
Most of these "feature" manifolds are stuff found with PCA in low dimensional. To have an edge with respect with this manual methos we should be able to find stuff in a different regime for example high dimensional manifolds.
I believe looking for general features in the wilde is doomed and should not be pursued so I think I would prefer to focus on specific dataset or specific task an hopefully find a mechanism explained by this feature manifold. 
Both for high-dim features and for mechanism discovery the biggest issue is that I have no idea on how to investigate a manifold if not via visual inspection. 