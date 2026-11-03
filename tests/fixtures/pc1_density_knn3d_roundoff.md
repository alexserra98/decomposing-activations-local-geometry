# kNN quadrature regression fixture

`pc1_density_knn3d_roundoff.pt` contains 1,500 reference points in three PCA
coordinates, one center/unit direction, an interval half-length and k=20.
It records a deterministic neighbour-switching integration case from the
temperature/layer-28 density pilot (component 3, direction 5 including PC1
at index 0). It contains no model weights or prompt strings.

With 16 initial uniform subdivisions, QUADPACK reports roundoff and estimates
relative error 1.5865e-4. Explicitly subdividing at every 201-position pilot
sample retains the 1e-4 tolerance. The test independently checks its integral
using direct Euclidean query distances and a 50,001-position trapezoidal grid.
