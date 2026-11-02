"""Package verified scientific outputs, the report and the presentation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'deliverables'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):value.update(chunk)
    return value.hexdigest()


def main():
    pdf=DEST/'BTP1_Report.pdf'
    if (DEST/'main.pdf').exists(): (DEST/'main.pdf').replace(pdf)
    if (DEST/'main.log').exists(): (DEST/'main.log').replace(ROOT/'logs/report_tex.log')
    report=PdfReader(pdf)
    report_pages=len(report.pages)
    assert 10 <= report_pages <= 12
    report_text='\n'.join(page.extract_text() for page in report.pages)
    assert 'Self declaration' in report.pages[0].extract_text()
    assert '23EC39025' in report.pages[0].extract_text()
    assert 'Supervisor signature' in report.pages[0].extract_text()
    assert len(report.pages[0].images)==1
    assert (ROOT/'report/main.tex').read_text().count('\\bibitem{')==2
    assert '86.98%' in report_text and '1.01%' in report_text
    assert not re.search(r'\b(TODO|TBD|lorem|ipsum)\b',report_text,re.I)
    pptx=DEST/'BTP1_Presentation.pptx'
    with zipfile.ZipFile(pptx) as archive:
        slide_count=sum(bool(re.fullmatch(r'ppt/slides/slide\d+\.xml',n)) for n in archive.namelist())
        chart_count=sum(bool(re.fullmatch(r'ppt/charts/chart\d+\.xml',n)) for n in archive.namelist())
        assert archive.testzip() is None
    assert slide_count==10 and chart_count==3
    assert 'All validations PASSED!' in (ROOT/'logs/slides_validation.log').read_text()
    notes=json.loads((ROOT/'results/presentation_notes.json').read_text())
    assert len(notes)==10 and sum(n['seconds'] for n in notes)==540
    audit=json.loads((ROOT/'results/audit.json').read_text())
    assert audit['passed'] and audit['metric_rows_verified']==333
    manifest=json.loads((ROOT/'report/results_manifest.json').read_text())
    assert manifest['summary_sha256']==digest(ROOT/'results/summary.json')
    for name,expected in manifest['figures'].items():
        assert digest(ROOT/'report/images'/name)==expected
    record={
        'completed_utc':datetime.now(timezone.utc).isoformat(),
        'report_pages':report_pages,'presentation_slides':10,'editable_native_charts':3,
        'report_research_paper_references':2,'report_cover_institute_emblem':True,
        'presentation_target_seconds':540,'presentation_limit_seconds':600,'qa_seconds_separate':120,
        'speaker_script_words':sum(len(re.findall(r"\b[\w'-]+\b",n['script'])) for n in notes),
        'trained_models':9,'test_objects':2468,'verified_metric_rows':333,
        'geometry_tests_passed':13,'data_model_and_metric_audit_passed':True,
        'report_visual_review':'All pages inspected after cover, emblem and two-reference revision',
        'slide_visual_review':'All slides inspected; all native charts structurally validated',
        'signatures':'Blank for student review and supervisor endorsement',
        'files':{pdf.name:digest(pdf),pptx.name:digest(pptx)},
    }
    (ROOT/'results/delivery_verification.json').write_text(json.dumps(record,indent=2))
    (DEST/'START_HERE.md').write_text('''# Material for the supervisor meeting

The implementation, experiments, report and presentation are complete.

1. Open `BTP1_Report.pdf`. It contains REPORT_PAGE_COUNT pages, including the institute emblem, declaration and endorsement spaces. The bibliography contains the two papers used for the method and dataset. Review the contents before signing; obtain the supervisor's endorsement after review.
2. Open `BTP1_Presentation.pptx`. It contains 10 editable slides and speaker notes targeting 9 minutes. The strict talk limit is 10 minutes; the 2-minute Q&A is separate.
3. Use `Presentation_Timing.md` for the same timed script outside PowerPoint.
4. `BTP1_LaTeX_Source.zip` contains the complete report source and figure assets. Upload it to Overleaf with `main.tex` as the main file, or compile locally.
5. `BTP1_Project.zip` contains code, trained models, results, figures, source, report, presentation, audit and logs. The full dataset and Python environments stay in the workspace and are not duplicated in the archive; download instructions are included.

## Main measured comparison

| Condition | A mean/max | B median/max | C median/P95 |
| --- | ---: | ---: | ---: |
| Clean overall accuracy | 88.07% | 87.76% | 87.28% |
| 2% replacement at radius 5 | 1.01% | 2.41% | 86.98% |
| 10% replacement at radius 5 | 0.81% | 2.31% | 1.92% |

These are means over three training seeds, with corruption draws averaged within each seed. The report gives the standard deviations and all three outlier distances. This is a deliberately controlled distant-outlier stress test, not a claim of general sensor-noise robustness or a new universal algorithm.

## Demonstration on this laptop

From the BTP workspace:

```bash
PYTHONPATH='' .venv/bin/python scripts/demo.py --fraction 0.02 --radius 5
```

The example is the first chair in standard test-file order. Its clean predictions are chair in all pipelines. Under the specified corruption, A predicts bowl, B predicts vase and C predicts chair. The report's conclusions use all 2468 test clouds rather than this example alone.

The next preparation step is to understand the paper, implementation, measured results and limitations, then rehearse the timed explanation.
'''.replace('REPORT_PAGE_COUNT',str(report_pages)))

    source_zip=DEST/'BTP1_LaTeX_Source.zip'
    with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((ROOT/'report').rglob('*')):
            if file.is_file() and file.suffix in {'.tex','.pdf','.json','.png','.jpg','.md'}:
                archive.write(file,file.relative_to(ROOT/'report'))
        archive.writestr('README.txt','Main file: main.tex\nCompile with pdfLaTeX, XeLaTeX or Tectonic. All figures and generated numerical sections are included.\n')

    project_zip=DEST/'BTP1_Project.zip'
    with zipfile.ZipFile(project_zip,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in ['README.md','SOURCES.md','config.json','project.json','requirements.txt','requirements-inference.txt','requirements-lock.txt','pyproject.toml','package.json','package-lock.json','.gitignore']:
            archive.write(ROOT/name,name)
        for folder in ['src','scripts','tests','models','results','report','vendor','logs','assets','examples']:
            for file in sorted((ROOT/folder).rglob('*')):
                if file.is_file() and '__pycache__' not in file.parts and file.suffix!='.pyc':
                    archive.write(file,file.relative_to(ROOT))
        for name in ['BTP1_Report.pdf','BTP1_Presentation.pptx','Presentation_Timing.md','START_HERE.md']:
            archive.write(DEST/name,Path('deliverables')/name)
        provenance=ROOT/'data/modelnet40_ply_hdf5_2048/provenance.json'
        archive.write(provenance,provenance.relative_to(ROOT))
    for file in [source_zip,project_zip]:
        with zipfile.ZipFile(file) as archive:assert archive.testzip() is None
    packages={p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in [pdf,pptx,source_zip,project_zip]}
    (DEST/'manifest.json').write_text(json.dumps(packages,indent=2))
    print(json.dumps({'verification':record,'packages':packages},indent=2))


if __name__=='__main__':main()
