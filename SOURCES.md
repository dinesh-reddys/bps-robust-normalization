# Sources and implementation provenance

1. Sergey Prokudin, Christoph Lassner and Javier Romero. **Efficient Learning on Point Clouds With Basis Point Sets**. ICCV, 2019, pp. 4332–4341. https://arxiv.org/abs/1908.09186
   - Source of the BPS representation and BPS–MLP classification architecture.
   - Supplement section 4 discusses outlier-sensitive normalization in mesh registration and suggests robust normalization or synthetic noise. This study credits that motivation and evaluates a specific preprocessing comparison for classification. It does not claim that robust normalization is a new general idea.
   - Author code: https://github.com/sergeyprokudin/bps
   - Inspected commit: `26788bceee94950b884a5f5c481aafee3b4fd4dc`.
   - The MLP follows `bps_demos/train_modelnet_mlp.py`: input BatchNorm; two 512-unit Linear/ReLU/BatchNorm/Dropout blocks; 40 logits; dropout probability 0.8; Adam. Our implementation uses validation selection, early stopping, repeated seeds and correct cross-entropy evaluation. It is a baseline adaptation, not an exact reproduction of every paper setting.
   - BPS geometry is independently implemented with scipy cKDTree. The shared random ball basis follows the author's sampling order, rounded to float32. Source code licence: MIT-0.
2. Charles R. Qi, Hao Su, Kaichun Mo and Leonidas J. Guibas. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**. CVPR, 2017, pp. 652–660. https://arxiv.org/abs/1612.00593
   - Related work on learning directly from unordered point sets. PointNet is discussed, not trained or experimentally compared here.
3. Jiawei Ren, Liang Pan and Ziwei Liu. **Benchmarking and Analyzing Point Cloud Classification under Corruptions**. ICML, PMLR 162, 2022, pp. 18559–18575. https://proceedings.mlr.press/v162/ren22c.html
   - Related work establishing controlled corruption evaluation. Our spherical outlier replacement is a custom diagnostic protocol, not the official PointCloud-C benchmark.
4. Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang and Jianxiong Xiao. **3D ShapeNets: A Deep Representation for Volumetric Shapes**. CVPR, 2015, pp. 1912–1920. https://3dshapenets.cs.princeton.edu/
   - Source of ModelNet. We use the 2048-point HDF5 conversion distributed for PointNet, not the volumetric model.
5. Hang Zhou, Kejiang Chen, Weiming Zhang, Han Fang, Wenbo Zhou and Nenghai Yu. **DUP-Net: Denoiser and Upsampler Network for 3D Adversarial Point Clouds Defense**. ICCV, 2019, pp. 1961–1970. https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_DUP-Net_Denoiser_and_Upsampler_Network_for_3D_Adversarial_Point_Clouds_ICCV_2019_paper.html
   - Related work and a future filtering comparison. No claim that normalization removes outliers or provides an adversarial defence.

## Dataset

- Original processed-data URL: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
- Public ETH research mirror used: https://share.phys.ethz.ch/~gseg/Predator/data/modelnet40_ply_hdf5_2048/
- `data/modelnet40_ply_hdf5_2048/provenance.json` records per-file URLs, byte counts, SHA-256 hashes and retrieval time.
- Official train/test membership is retained. Only the official training split is partitioned for validation.
- The downloaded PointNet HDF5 conversion contains 9840 training clouds and 2468 test clouds. These counts are verified directly from HDF5 shapes and reported as such.

## Assistance

The project was prepared with Codex assistance for implementation, execution, analysis and document drafting. The user should review the code, results and declaration before signing or submitting the report. No supervisor signature is generated.
