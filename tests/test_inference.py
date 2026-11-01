from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from btp.inference import BPSClassifier,load_points
from btp.core import hash_file,replace_outliers


@pytest.fixture(scope='module')
def classifier():
    return BPSClassifier('C',13,'cpu')


def test_bundled_chair_identity_and_prediction(classifier):
    metadata=json.loads((ROOT/'examples/chair.json').read_text())
    assert hash_file(ROOT/'examples/chair.npy')==metadata['sample_sha256']
    result=classifier.predict(load_points(ROOT/'examples/chair.npy'))
    assert result['category']=='chair' and result['points']==2048


def test_matched_chair_example_matches_saved_experiment():
    clean=load_points(ROOT/'examples/chair.npy')
    noisy,_=replace_outliers(clean,.02,5,101,3)
    expected=json.loads((ROOT/'results/demo.json').read_text())['predictions']
    for pipeline in 'ABC':
        assert BPSClassifier(pipeline).predict(noisy)['category']==expected[pipeline]['corrupted']


@pytest.mark.parametrize('points',[np.zeros((10,4)),np.empty((0,3)),np.full((10,3),np.nan)])
def test_invalid_point_clouds_fail_clearly(classifier,points):
    with pytest.raises(ValueError):classifier.predict(points)


def test_npz_input_loader(tmp_path):
    path=tmp_path/'input.npz'
    points=np.arange(30,dtype=np.float32).reshape(10,3)
    np.savez(path,points=points)
    np.testing.assert_array_equal(load_points(path),points)


def test_cli_can_run_outside_repository(tmp_path):
    output=subprocess.run([sys.executable,str(ROOT/'scripts/predict.py'),'--input',str(ROOT/'examples/chair.npy'),'--device','cpu'],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(output.stdout)['category']=='chair'
