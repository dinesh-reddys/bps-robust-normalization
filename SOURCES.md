# Sources and implementation provenance

1. Sergey Prokudin, Christoph Lassner and Javier Romero. **Efficient Learning on Point Clouds With Basis Point Sets**. ICCV, 2019, pp. 4332–4341. https://arxiv.org/abs/1908.09186
   - Source of the BPS representation and BPS–MLP classification architecture.
   - Supplement section 4 discusses outlier-sensitive normalization in mesh registration and suggests robust normalization or synthetic noise. This study credits that motivation and evaluates a specific preprocessing comparison for classification. It does not claim that robust normalization is a new general idea.
   - Author code: https://github.com/sergeyprokudin/bps
   - Inspected commit: `26788bceee94950b884a5f5c481aafee3b4fd4dc`.
   - The MLP follows `bps_demos/train_modelnet_mlp.py`: input BatchNorm; two 512-unit Linear/ReLU/BatchNorm/Dropout blocks; 40 logits; dropout probability 0.8; Adam. Our implementation uses validation selection, early stopping, repeated seeds and correct cross-entropy evaluation. It is a baseline adaptation, not an exact reproduction of every paper setting.
   - BPS geometry is independently implemented with scipy cKDTree. The shared random ball basis follows the author's sampling order, rounded to float32. Source code licence: MIT-0.
2. Zhirong Wu, Shuran Song, Aditya Khosla, Fisher Yu, Linguang Zhang, Xiaoou Tang and Jianxiong Xiao. **3D ShapeNets: A Deep Representation for Volumetric Shapes**. CVPR, 2015, pp. 1912–1920. https://3dshapenets.cs.princeton.edu/
   - Source of ModelNet. We use its CAD-derived object dataset through the 2048-point HDF5 conversion, not the paper's volumetric model.

The report cites these two research papers because they directly supply the method and dataset. Its spherical outlier replacement is a custom diagnostic protocol, and its comparison is limited to preprocessing rules within BPS–MLP.

## Software and report format

- PointNet data-conversion distribution: https://github.com/charlesq34/pointnet. This credits the processed HDF5 dataset; no PointNet classifier is trained or compared.
- The cover adapts the MUST Lab, IIT Kharagpur M.Tech LaTeX template. The unmodified institute emblem comes from that archive. See [template provenance](report/TEMPLATE_SOURCES.md) for source links and adaptations to the supervisor's BTP-1 requirements.

## Dataset

- Original processed-data URL: https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip
- Public ETH research mirror used: https://share.phys.ethz.ch/~gseg/Predator/data/modelnet40_ply_hdf5_2048/
- `data/modelnet40_ply_hdf5_2048/provenance.json` records per-file URLs, byte counts, SHA-256 hashes and retrieval time.
- Official train/test membership is retained. Only the official training split is partitioned for validation.
- The downloaded PointNet HDF5 conversion contains 9840 training clouds and 2468 test clouds. These counts are verified directly from HDF5 shapes and reported as such.

## Assistance

The project was prepared with Codex assistance for implementation, execution, analysis and document drafting. The user should review the code, results and declaration before signing or submitting the report. No supervisor signature is generated.
