"""Audit dataset membership, checkpoints and every saved accuracy independently."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from btp.core import hash_file,load_dataset,normalize


def main():
    out=ROOT/'results'
    cfg=json.loads((ROOT/'config.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text())
    assert protocol['config_sha256']==hash_file(ROOT/'config.json')
    assert protocol['basis_sha256']==hash_file(out/'basis.npy')
    provenance_path=ROOT/'data/modelnet40_ply_hdf5_2048/provenance.json'
    assert protocol['dataset_provenance_sha256']==hash_file(provenance_path)
    provenance=json.loads(provenance_path.read_text())
    for record in provenance['files']:
        file=provenance_path.parent/record['file']
        assert file.stat().st_size==record['bytes'] and hash_file(file)==record['sha256']
    split=np.load(out/'split_indices.npz')
    assert len(np.intersect1d(split['train'],split['validation']))==0
    np.testing.assert_array_equal(np.sort(np.r_[split['train'],split['validation']]),np.arange(9840))
    xtr,ytr,xte,yte,names=load_dataset(ROOT)
    assert len(np.unique(ytr[split['train']]))==40 and len(np.unique(ytr[split['validation']]))==40
    assert len(np.unique(yte))==40
    ids={}
    for group in ['train','test']:
        all_ids=[]
        for file in sorted((ROOT/'data/modelnet40_ply_hdf5_2048').glob(f'ply_data_{group}_*_id2file.json')):
            value=json.loads(file.read_text())
            if not isinstance(value,list): raise ValueError('Unexpected object ID format')
            all_ids.extend(value)
        ids[group]=set(all_ids)
    assert not ids['train'].intersection(ids['test']), 'Mesh identity crosses official train/test split'
    model_checks=[]
    for p in 'ABC':
        for seed in cfg['training_seeds']:
            metadata=json.loads((ROOT/f'models/{p}_seed{seed}.json').read_text())
            assert metadata['checkpoint_sha256']==hash_file(ROOT/f'models/{p}_seed{seed}.pt')
            assert metadata['config_sha256']==hash_file(ROOT/'config.json')
            assert 1<=metadata['best_epoch']<=metadata['epochs_run']<=cfg['max_epochs']
            model_checks.append({'pipeline':p,'seed':seed,'checkpoint_hash_matches':True})
    with (out/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==333
    checks=0
    for file in sorted((out/'predictions').glob('*.npz')):
        data=np.load(file)
        np.testing.assert_array_equal(data['labels'],yte)
        records=json.loads(file.with_suffix('.json').read_text())['metrics']
        for row in records:
            position=cfg['training_seeds'].index(row['training_seed'])
            pred=data[row['pipeline']+'_predictions'][position]
            assert pred.shape==(2468,) and pred.min()>=0 and pred.max()<40
            accuracy=float(np.mean(pred==yte))
            mca=float(np.mean([np.mean(pred[yte==c]==c) for c in range(40)]))
            assert abs(accuracy-row['accuracy'])<1e-12
            assert abs(mca-row['mean_class_accuracy'])<1e-12
            matches=[r for r in rows if r['pipeline']==row['pipeline'] and int(r['training_seed'])==row['training_seed'] and float(r['outlier_fraction'])==row['outlier_fraction'] and float(r['outlier_radius'])==row['outlier_radius'] and int(r['corruption_seed'])==row['corruption_seed']]
            assert len(matches)==1
            assert abs(float(matches[0]['accuracy'])-accuracy)<1e-12
            checks+=1
    assert checks==333
    # Brute-force distances verify the spatial-tree feature implementation.
    basis=np.load(out/'basis.npy')
    for p in 'ABC':
        cloud=normalize(xte[0],p,cfg['percentile'])[0]
        brute=np.linalg.norm(basis[:16,None,:]-cloud[None,:,:],axis=2).min(axis=1)
        cached=np.load(ROOT/f'cache/clean_{p}.npz')['test'][0,:16]
        np.testing.assert_allclose(cached,brute,atol=2e-7)
    result={'passed':True,'metric_rows_verified':checks,'prediction_files':len(list((out/'predictions').glob('*.npz'))),
            'train_objects':len(xtr),'test_objects':len(xte),'train_validation_overlap':0,
            'official_mesh_identity_overlap':0,'spatial_tree_matches_brute_force':True,'dataset_file_hashes_verified':len(provenance['files']),
            'models':model_checks,'config_sha256':hash_file(ROOT/'config.json')}
    (out/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
