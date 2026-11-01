<!-- Slide number: 1 -->
Normalization for BPS based 3D object classification
A controlled study of distant outliers
SOMAVARAPU DINESH REDDY  |  23EC39025
Supervisor  Prof. Prabir Kumar Biswas
E&ECE  ·  IIT Kharagpur
![pipeline](pipeline)
1

### Notes:
Target 20 seconds.

My BTP study examines how normalization affects 3D object classification when point clouds contain distant outliers. I use the Basis Point Sets representation and a small multilayer perceptron, then compare three preprocessing rules.

<!-- Slide number: 2 -->
A distant point can change the object scale
Clean object
Radius about 1
One far point
Maximum radius grows
Maximum based scaling shrinks the genuine object relative to the fixed BPS basis.
Question: can a robust scale estimate preserve classification accuracy?
Motivation: Prokudin et al., ICCV 2019, supplementary discussion of normalization sensitivity
2

### Notes:
Target 50 seconds.

The original preprocessing subtracts the mean of the cloud and divides coordinates by the largest distance from that centre. If an outlier lies five times farther away, the genuine object can become approximately five times smaller after normalization. BPS measures distances from fixed reference locations, so this changes many features together. The authors already identify this normalization sensitivity in their supplementary discussion. My question is whether a specific robust normalization rule reduces the classification damage, and what clean performance or failure limits accompany it.

<!-- Slide number: 3 -->
BPS turns points into a fixed feature vector
![bps explanation](bps_explanation)
512 fixed reference points
For each reference point, record the nearest distance to the object.
Point order does not change the minimum distance.
Prokudin, Lassner and Romero, ICCV 2019  |  Diagram is a 2D illustration; the implementation uses 3D coordinates
3

### Notes:
Target 60 seconds.

A point cloud is an unordered collection of xyz coordinates. BPS samples one fixed set of reference points and uses exactly that set for every object. For each reference point, I find the nearest cloud point and store the Euclidean distance. With 512 reference points, every cloud becomes a vector of 512 numbers. The diagram is a two-dimensional illustration, but the implementation is three-dimensional. Reordering the object points does not change the minimum over the set, so the representation is permutation invariant. It is not generally rotation invariant because the reference locations remain fixed.

<!-- Slide number: 4 -->
A small MLP predicts the object category
512 distances
512 hidden units
512 hidden units
40 class scores
Weighted sums → ReLU → batch normalization → dropout
Train with cross entropy and Adam
Predict the class with the largest output score
Architecture adapted from the public BPS MLP example; identical dimensions and training settings across A, B and C
4

### Notes:
Target 55 seconds.

The classifier extends the perceptron idea using two hidden layers. Each linear layer computes weighted sums, and ReLU introduces a nonlinearity. Batch normalization and dropout follow the author example. The output has 40 scores, one for each category. Cross entropy measures classification error during training, backpropagation computes gradients, and Adam updates the weights. At inference, dropout is disabled and the largest output score determines the category. The BPS reference points are fixed rather than learned. All three pipelines have the same classifier architecture; each is trained using its own preprocessing.

<!-- Slide number: 5 -->
Three pipelines isolate centring and scaling

| Pipeline | Centre | Scale |
| --- | --- | --- |
| A Original | Mean | Maximum radius |
| B Centring control | Coordinate wise median | Maximum radius |
| C Robust variant | Coordinate wise median | 95th percentile radius |

A → B tests centring
B → C tests scaling
All points are retained.
The percentile rule does not remove outliers.
All models use the same saved BPS basis; the 95th percentile is fixed before final test evaluation
5

### Notes:
Target 55 seconds.

Pipeline A uses the original mean centre and maximum radius. B changes only the centre to the coordinate-wise median. C also replaces maximum radius by the 95th percentile of radii. A versus B tells us whether centring alone helps. B versus C isolates the scale change. The percentile can ignore a small upper tail of extreme radii when estimating the object scale. However, all points remain in the cloud. This is a preprocessing investigation motivated by the source paper, not a claim that robust normalization is a new general algorithm.

<!-- Slide number: 6 -->
Models are selected before test evaluation
7,872
1,968
2,468
Fitting objects
Validation objects
Test objects
ModelNet40  ·  2048 points per object  ·  40 categories
3 training seeds × 3 corruption draws
Replacement: 1%, 2%, 5%, 10%  |  Distances: 3×, 5×, 10×
PointNet HDF5 conversion: 9840 training clouds; validation is drawn only from that split. Matched corruptions across pipelines.
6

### Notes:
Target 60 seconds.

The processed ModelNet40 archive contains 9840 training clouds and 2468 test clouds. I reserve 1968 training objects for validation and fit on 7872. Validation chooses checkpoints; the test set is used only after training. Each pipeline is trained with three seeds. Evaluation replaces selected points with points on a distant spherical shell, keeping the total point count fixed. I test four replacement fractions and three distances, with three corruption draws. The same corrupted objects are supplied to all pipelines. This is a controlled diagnostic protocol, not the official PointCloud-C benchmark or realistic sensor noise.

<!-- Slide number: 7 -->
What happens to classification accuracy

### Chart

| Category | A | B | C |
| --- | --- | --- | --- |
| Clean | 88.07401404646137 | 87.76337115072934 | 87.2771474878444 |
| 2% at radius 5 | 1.012965964343598 | 2.408607959661444 | 86.98451287592293 |
| 10% at radius 5 | 0.8103727714748784 | 2.3095623987034037 | 1.917882225823879 |

+85.97 pp
C minus A
2% replacement
Radius 5
Measured test means; corruption draws averaged within training seeds. Report contains SD across seeds and the full distance sweep.
7

### Notes:
Target 75 seconds.

The clean accuracies are 88.07 percent for A, 87.76 for B and 87.28 for C. At two-percent replacement and radius five, the corresponding accuracies are 1.01, 2.41 and 86.98 percent. C changes accuracy relative to A by 85.97 percentage points. The right group shows the ten-percent condition, which is important for interpreting the method's limit. These bars show means; the report also gives training-seed standard deviations and the complete radius sweep. The comparison does not establish performance against other architectures or real sensor corruptions.

<!-- Slide number: 8 -->
Geometry explains the prediction changes
![mechanism](mechanism)
Radius-five diagnostic. Main-object size is measured on retained inliers; BPS feature drift is measured before classification.
8

### Notes:
Target 65 seconds.

The geometric diagnostics help explain the accuracy results. On the left, I measure the size of the genuine retained object after normalization relative to its clean size. A value close to one means that its scale is preserved. With a far outlier, maximum-based scaling can produce a value near one fifth at radius five. The right plot measures how much the 512-dimensional distance vector changes. Together these measurements connect preprocessing to the classifier input. They do not prove that every changed feature causes an error, but they support the proposed scale mechanism independently of the predicted labels.

<!-- Slide number: 9 -->
Percentile scaling has an operating limit
Heavy contamination can enter the percentile itself
10% extreme points
95th percentile
can be an outlier radius
Object shrinks
Nearby clutter can also distort nearest distances.
Normalization does not remove those points.
Observed high-contamination results are reported alongside successful conditions; no universal robustness threshold is claimed.
9

### Notes:
Target 60 seconds.

The percentile rule is not immune to contamination. If enough points have extreme radii, the 95th percentile can itself be determined by outliers, and the genuine object can shrink again. The exact transition depends on finite sampling and the radial distribution. There is also a second mechanism: an outlier near a reference point can directly become its nearest neighbour and change that feature. Normalization does not remove it. My distant-shell experiment mainly investigates scale sensitivity. Nearby clutter, Gaussian jitter, missing regions and real scans remain separate tests.

<!-- Slide number: 10 -->
Findings and next experiments
A controlled preprocessing change can be evaluated with both predictions and geometry.
Completed  BPS classifier · A/B/C ablation · repeated-seed evaluation
Next  Outlier filtering · noise augmentation · broader corruptions
References: Prokudin et al., ICCV 2019; Qi et al., CVPR 2017;
Ren et al., ICML 2022; Wu et al., CVPR 2015.
10

### Notes:
Target 40 seconds.

The completed work is a BPS classifier with a controlled normalization ablation and repeated evaluation. Its contribution is the implemented comparison and explanation of an acknowledged failure mechanism. The results need an operating range and a clean-accuracy comparison. The next stage is an outlier-filtering baseline, followed by noise augmentation and broader corruptions. Thank you.
