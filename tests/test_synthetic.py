import numpy as np
import pytest

from dxa_sandbox import evaluation, pipeline, synthetic
from dxa_sandbox.config import series_uid_by_mask
from dxa_sandbox.geometry import image_center, rotate_image, rotate_points
from dxa_sandbox.reading import read

IMPLEMENTED = [
    ("spine", "incomplete_coverage"), ("spine", "positioning"), ("spine", "foreign_body_artifact"),
    ("spine", "acquisition_parameters"), ("spine", "vertebra_labeling"), ("spine", "roi_placement"),
    ("hip", "foreign_body_artifact"), ("hip", "acquisition_parameters"),
]


def _qc_defects(case, tmp_path, name):
    path = tmp_path / f"{name}.dcm"
    synthetic.write_case(case, synthetic.render_case(case), path, name)
    return pipeline.run_qc(read(path)).defects - {"dicom_data"}


def test_rotate_points_matches_rotate_image():
    image = np.zeros((101, 81), np.float32)
    image[18:23, 58:63] = 1
    rotated = rotate_image(image, 30)
    ys, xs = np.nonzero(rotated > 0.3)
    (x, y), = rotate_points([(60, 20)], 30, image_center(image.shape))
    assert abs(xs.mean() - x) < 1 and abs(ys.mean() - y) < 1


def test_series_uid_mask():
    assert series_uid_by_mask("1.2.3", 1000) == "1.2.3.1000.1.1"
    long_uid = "1." + "2" * 70
    uid = series_uid_by_mask(long_uid, 1000, add_id=2)
    assert len(uid) <= 64 and uid.endswith(".1000.2.1")


def test_label_shift_moves_roi_up_one_vertebra():
    case = synthetic.generate_case("spine", np.random.default_rng(0), [])
    top_before = min(y for _, y in case.markup["roi"])
    synthetic.shift_vertebra_labels(case, -1)
    top_after = min(y for _, y in case.markup["roi"])
    t12_top, l1_top = (min(y for _, y in case.anatomy[n]) for n in ("T12", "L1"))
    assert top_after < top_before and abs((top_before - top_after) - (l1_top - t12_top)) < 1


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind", ["spine", "hip"])
def test_clean_cases_have_no_defects(kind, seed, tmp_path):
    case = synthetic.generate_case(kind, np.random.default_rng(seed), [])
    assert _qc_defects(case, tmp_path, f"clean_{kind}_{seed}") == set()


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("kind,defect", IMPLEMENTED)
def test_implemented_checks_detect_defect(kind, defect, seed, tmp_path):
    case = synthetic.generate_case(kind, np.random.default_rng(seed), [defect])
    assert defect in _qc_defects(case, tmp_path, f"{kind}_{defect}_{seed}")


def test_benchmark_end_to_end(tmp_path):
    labels = synthetic.generate_dataset(tmp_path / "data", n=20, seed=1)
    assert len(labels) == 20 and (tmp_path / "data" / "labels.jsonl").exists()
    assert all("geometry" in label for label in labels)
    pipeline.process_path(tmp_path / "data" / "cases", tmp_path / "qc")
    metrics = evaluation.evaluate(evaluation.load_labels(tmp_path / "data" / "labels.jsonl"),
                                  evaluation.collect_predictions(tmp_path / "qc"))
    assert metrics["missing_predictions"] == 0
    assert set(metrics["per_defect"]) == set(evaluation.EVAL_DEFECTS)
    assert "чувств." in evaluation.format_report(metrics)
