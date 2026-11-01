# Presentation timing

Target: 9 minutes. Strict limit: 10 minutes. Q&A: 2 minutes, separate.

## Slide 1 — 20 seconds

My BTP study examines how normalization affects 3D object classification when point clouds contain distant outliers. I use the Basis Point Sets representation and a small multilayer perceptron, then compare three preprocessing rules.

## Slide 2 — 50 seconds

The original preprocessing subtracts the mean of the cloud and divides coordinates by the largest distance from that centre. If an outlier lies five times farther away, the genuine object can become approximately five times smaller after normalization. BPS measures distances from fixed reference locations, so this changes many features together. The authors already identify this normalization sensitivity in their supplementary discussion. My question is whether a specific robust normalization rule reduces the classification damage, and what clean performance or failure limits accompany it.

## Slide 3 — 60 seconds

A point cloud is an unordered collection of xyz coordinates. BPS samples one fixed set of reference points and uses exactly that set for every object. For each reference point, I find the nearest cloud point and store the Euclidean distance. With 512 reference points, every cloud becomes a vector of 512 numbers. The diagram is a two-dimensional illustration, but the implementation is three-dimensional. Reordering the object points does not change the minimum over the set, so the representation is permutation invariant. It is not generally rotation invariant because the reference locations remain fixed.

## Slide 4 — 55 seconds

The classifier extends the perceptron idea using two hidden layers. Each linear layer computes weighted sums, and ReLU introduces a nonlinearity. Batch normalization and dropout follow the author example. The output has 40 scores, one for each category. Cross entropy measures classification error during training, backpropagation computes gradients, and Adam updates the weights. At inference, dropout is disabled and the largest output score determines the category. The BPS reference points are fixed rather than learned. All three pipelines have the same classifier architecture; each is trained using its own preprocessing.

## Slide 5 — 55 seconds

Pipeline A uses the original mean centre and maximum radius. B changes only the centre to the coordinate-wise median. C also replaces maximum radius by the 95th percentile of radii. A versus B tells us whether centring alone helps. B versus C isolates the scale change. The percentile can ignore a small upper tail of extreme radii when estimating the object scale. However, all points remain in the cloud. This is a preprocessing investigation motivated by the source paper, not a claim that robust normalization is a new general algorithm.

## Slide 6 — 60 seconds

The processed ModelNet40 archive contains 9840 training clouds and 2468 test clouds. I reserve 1968 training objects for validation and fit on 7872. Validation chooses checkpoints; the test set is used only after training. Each pipeline is trained with three seeds. Evaluation replaces selected points with points on a distant spherical shell, keeping the total point count fixed. I test four replacement fractions and three distances, with three corruption draws. The same corrupted objects are supplied to all pipelines. This is a controlled diagnostic protocol, not the official PointCloud-C benchmark or realistic sensor noise.

## Slide 7 — 75 seconds

The clean accuracies are 88.07 percent for A, 87.76 for B and 87.28 for C. At two-percent replacement and radius five, the corresponding accuracies are 1.01, 2.41 and 86.98 percent. C changes accuracy relative to A by 85.97 percentage points. The right group shows the ten-percent condition, which is important for interpreting the method's limit. These bars show means; the report also gives training-seed standard deviations and the complete radius sweep. The comparison does not establish performance against other architectures or real sensor corruptions.

## Slide 8 — 65 seconds

The geometric diagnostics help explain the accuracy results. On the left, I measure the size of the genuine retained object after normalization relative to its clean size. A value close to one means that its scale is preserved. With a far outlier, maximum-based scaling can produce a value near one fifth at radius five. The right plot measures how much the 512-dimensional distance vector changes. Together these measurements connect preprocessing to the classifier input. They do not prove that every changed feature causes an error, but they support the proposed scale mechanism independently of the predicted labels.

## Slide 9 — 60 seconds

The percentile rule is not immune to contamination. If enough points have extreme radii, the 95th percentile can itself be determined by outliers, and the genuine object can shrink again. The exact transition depends on finite sampling and the radial distribution. There is also a second mechanism: an outlier near a reference point can directly become its nearest neighbour and change that feature. Normalization does not remove it. My distant-shell experiment mainly investigates scale sensitivity. Nearby clutter, Gaussian jitter, missing regions and real scans remain separate tests.

## Slide 10 — 40 seconds

The main result is conditional. At radius five, C retains 86.98 percent accuracy with two-percent replacement, but falls to 1.92 percent with ten-percent replacement. Its clean accuracy is about 0.8 percentage points below A. The contribution is the implemented comparison and explanation of an acknowledged failure mechanism. The next stage is outlier filtering, followed by augmentation and broader corruptions. Thank you.
