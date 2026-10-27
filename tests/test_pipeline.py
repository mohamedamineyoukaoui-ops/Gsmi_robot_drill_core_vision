import cv2
import numpy as np
import pytest

from pipeline import pieces, rqd
from pipeline.cli import demo, main
from pipeline.cracks import apply_line_corrections
from pipeline.image_processing.stitching import assemble_grid, parse_grid_position, stitch_blind
from pipeline.lithology_fusion import decide
from pipeline.zones import reclassify_with_cracks


def test_known_grid_stitching_rebuilds_the_scene():
    scene = np.random.default_rng(4).integers(15, 230, (100, 300, 3), dtype=np.uint8)
    result = assemble_grid([scene[:, :190], scene[:, 110:]], [(0, 0), (0, 1)])
    np.testing.assert_array_equal(result, scene)


def test_grid_position_is_read_from_file_name():
    assert parse_grid_position("capture_r2c10_001.png") == (2, 10)
    assert parse_grid_position("image.png") is None
    with pytest.raises(ValueError):
        stitch_blind([])


def test_rqd_counts_overlapping_pieces_once():
    found = [dict(y_start=0, y_end=120, counts_for_rqd=True), dict(y_start=80, y_end=200, counts_for_rqd=True)]
    result = rqd.rqd_from_pieces(found, core_length_cm=30, px_per_cm=10)
    assert result["rqd_length_cm"] == 20
    assert result["rqd_percent"] == 66.67


def test_empty_mask_is_not_an_intact_piece():
    assert pieces.extract_pieces_cc(np.zeros((100, 20), np.uint8), px_per_cm=10) == []


def test_contour_method_splits_at_cracks_and_applies_10cm_rule():
    mask = np.zeros((600, 60), np.uint8)
    cv2.rectangle(mask, (0, 0), (59, 599), 255, 2)
    for y in (150, 190, 420):
        cv2.line(mask, (0, y), (59, y), 255, 3)
    found = pieces.extract_pieces_contour(mask, None, (0, 600), px_per_cm=10)
    assert len(found) == 4
    assert [p["counts_for_rqd"] for p in found] == [True, False, True, True]


def test_confident_rubble_zone_vetoes_a_long_piece():
    found = [pieces._piece(0, 200, px_per_cm=10)]                 # 20 cm
    zones = [{"zone_class": "rubble", "probs": {"rubble": 0.9}, "y_start": 0, "y_end": 150}]
    vetoes = pieces.apply_zone_rules(found, zones)
    assert not found[0]["counts_for_rqd"] and found[0]["is_rubble"]
    assert vetoes[0]["class"] == "rubble"
    # an unsure classifier (p < 0.70) does not overrule the piece
    found = [pieces._piece(0, 200, px_per_cm=10)]
    pieces.apply_zone_rules(found, [dict(zones[0], probs={"rubble": 0.6})])
    assert found[0]["counts_for_rqd"]


def test_run_length_becomes_the_denominator():
    cols = [{"core_length_cm": 80.0, "rqd_length_cm": 60.0}]
    assert rqd.overall_rqd(cols)["rqd_percent"] == 75.0               # 60 / 80 detected
    with_run = rqd.overall_rqd(cols, run_length_cm=100.0)               # 20 cm of core lost
    assert with_run["rqd_percent"] == 60.0 and with_run["core_recovery_percent"] == 80.0


def test_piece_cut_by_column_boundary_is_counted_whole():
    # 6 cm at the bottom of column 1 + 6 cm at the top of column 2 = one 12 cm piece
    cols = [
        {"height_px": 100, "core_length_cm": 10.0,
         "pieces": [dict(y_start=40, y_end=100, length_cm=6.0, counts_for_rqd=False)]},
        {"height_px": 100, "core_length_cm": 10.0,
         "pieces": [dict(y_start=0, y_end=60, length_cm=6.0, counts_for_rqd=False)]},
    ]
    pieces.merge_across_columns(cols)
    assert all(c["pieces"][0]["counts_for_rqd"] for c in cols)
    assert rqd.fracture_frequency(cols)["fracture_count"] == 0


def _box(y1, y2, conf):
    return {"x1": 0, "x2": 100, "y1": y1, "y2": y2, "confidence": conf}


def _column(height):
    return {"x_start": 0, "x_end": 100, "y_start": 0, "y_end": height}


def test_duplicate_detector_box_keeps_the_more_confident_one():
    found = pieces.pieces_from_boxes([_box(100, 400, .5), _box(110, 405, .9)], _column(500), px_per_cm=10)
    assert [(p["y_start"], p["y_end"]) for p in found] == [(110, 405)]


def _spans(found):
    return [(p["y_start"], p["y_end"]) for p in found]


def test_partial_overlap_is_split_at_its_middle():
    out = pieces.resolve_overlaps([{"y_start": 0, "y_end": 300, "confidence": .8},
                                   {"y_start": 280, "y_end": 600, "confidence": .8}], px_per_cm=10)
    assert [(p["y_start"], p["y_end"]) for p in out] == [(0, 290), (290, 600)]
    assert all(p["counts_for_rqd"] for p in out)


def test_long_weak_box_over_confident_pieces_keeps_the_pieces():
    # the same depth reported twice: a weak box over the whole column and two confident pieces
    boxes = [_box(0, 900, .3), _box(0, 400, .8), _box(400, 900, .7)]
    assert _spans(pieces.pieces_from_boxes(boxes, _column(900), px_per_cm=10)) == [(0, 400), (400, 900)]


def test_stacked_pieces_never_count_a_depth_twice():
    boxes = [_box(0, 210, .9), _box(200, 520, .8), _box(505, 800, .85)]
    found = pieces.pieces_from_boxes(boxes, _column(800), px_per_cm=10)
    assert _spans(found) == [(0, 205), (205, 512), (512, 800)]


def test_single_long_detector_box_is_kept():
    found = pieces.pieces_from_boxes([_box(0, 900, .4)], _column(900), px_per_cm=10)
    assert [(p["y_start"], p["y_end"]) for p in found] == [(0, 900)]


def test_manual_crack_is_drawn_in_image_coordinates():
    mask = np.zeros((30, 40), np.uint8)
    apply_line_corrections(mask, [[(5, 10), (30, 10)]])
    assert mask[10, 15] > 0
    assert mask[25, 15] == 0


def test_continuous_crack_turns_crushed_zone_intact():
    segments = [{"type": "crushed", "y_start": 50, "y_end": 80}]
    assert reclassify_with_cracks(segments, [{"y_top": 40, "y_bottom": 60}])[0]["type"] == "intact"
    assert reclassify_with_cracks(segments, [{"y_top": 55, "y_bottom": 70}])[0]["type"] == "crushed"


def test_empty_tray_is_not_called_a_rock():
    result = decide({"no_rock": 0.99, "limestone": 0.01}, None)
    assert result.target is None
    assert result.confidence == 0.99


def test_xrf_overrules_image_and_raises_conflict():
    # image: limestone; chemistry: flint (SiO2 81.8 %)
    result = decide({"limestone": 0.9, "no_rock": 0.1}, {"Si": 38.2, "Ca": 5.0, "P": 1.56, "Fe": 0.75})
    assert result.target in ("flint", "phosphate_flint")
    assert result.conflict


def test_demo_runs_without_weights(tmp_path):
    data = demo(tmp_path)
    assert data["stitching_exact"]
    assert (tmp_path / "pieces.png").is_file() and (tmp_path / "results.json").is_file()


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
def test_cli_rejects_invalid_scale(tmp_path, scale, capsys):
    with pytest.raises(SystemExit) as error:
        main(["analyze", "unused.png", "--pixels-per-cm", scale, "--output", str(tmp_path)])
    assert error.value.code == 2
    assert "finite and positive" in capsys.readouterr().err


def test_cli_reports_missing_weights_before_reading_the_image(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["analyze", "unused.png", "--weights", str(tmp_path), "--pixels-per-cm", "10",
              "--output", str(tmp_path / "out")])
    assert error.value.code == 2
    assert "Missing local model assets" in capsys.readouterr().err
