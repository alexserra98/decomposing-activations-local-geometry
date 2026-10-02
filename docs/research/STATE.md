# Research State (updated 2026-09-29)

> **Kind:** Research state · **Status:** Current as of 2026-09-29 · **Use when:**
> Establishing the current research direction, findings, and immediate goals.

I have assessed that MFA actually beat kmeans on noisy datasets especially for low-dimensional manifolds. Unfortunately the metric I used - tangent containment - is not very informative and it indicate more a necessary rather than sufficient condition for good tiling. 
I have updated tangent containment which now measure the maximum tangent alignement for each subset of dimension r and of the first q_k PCs of the covariance matrix and the r-dimensiona base of the manifold tangent space. I also created a new  adjusted tangent alignment that penalise when q_k < r. 
Under these new set metric it was clear that MFA is better only in the noisy setting but fail to beat kmeans in basically any other setting and metric. 
I have update MFA to use EM instead of Adam (sgd) and now I have finally something the is comparable to kmeans in low-noise regime and better in high-noise regime.
The only problems left to solve are:
- [ ] Both Kmeans and MFA-em fail to learn the tangent dimension of the swiss roll and learn the ambient dim instead (although the tangent alignment is good)
- [ ] Both Kmeans and MFA-em fail to tile of the 12d torus and 10d hypersphere because it learn the ambient dim instead insted of tangent dim and becuase tangent alignment is not high 