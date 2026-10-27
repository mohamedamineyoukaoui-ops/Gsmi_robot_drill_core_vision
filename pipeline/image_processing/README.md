# Image processing (no AI)

[Project overview](../../README.md) · [Pipeline guide](../README.md)

The scanning robot photographs a tray in a grid of overlapping close-up shots, 120 for a full tray. Before any model runs, they must become one image with a known scale in pixels per centimetre. Every length in the pipeline depends on that scale: the 10 cm RQD threshold is applied in pixels.

## Run the stitcher

Stitching needs only the base package dependencies; no model weights are required. From the project root, with the package installed in your environment:

```bash
python -m pipeline.cli stitch scans/tray_01/ --output tray.png
```

The folder must contain at least two readable images. Supported extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif` and `.tiff`; images are read directly from that folder, without recursing into subfolders. Keep the output outside the input folder so it is not included in a later stitch.

The command writes `tray.png` and prints JSON with its path and `pixels_per_cm`. For a successful robot-position assembly, it also writes `tray_lithology.jpg`, with the same layout at higher resolution. Pass that file as `--lithology-image` during analysis. The Python entry point is `stitch_folder(folder)`, returning `(bgr_image, pixels_per_cm, hires_bgr_or_none)`.

## Stitching: [stitching.py](stitching.py)

Core trays defeat ordinary panorama stitching. Dividers, tray walls and core pieces look alike, so feature matching (SIFT, ORB) regularly matches the wrong place and drops or duplicates core. The stitcher therefore uses the strongest information available first:

![Stitching strategy: robot positions, then known grid, then blind stitching](../../docs/diagrams/stitching_strategy.png)

### 1. Robot positions

The robot records where each photo was taken in millimetres, so the layout is known and scale can be estimated from image overlap.

- **Scale and orientation by consensus.** For each of the 8 ways the camera image can be mirrored or rotated, the shift between neighbouring photos is found by template matching. The match scores of *all* neighbour pairs are summed before taking the peak, so a few wrong matches are outvoted. Shift ÷ robot step gives pixels per mm, and therefore pixels per cm with no marker.
- **Joint fine-tuning.** Each photo is placed at scale × robot position. Template matching may then move it by a few pixels: a least-squares problem over all photos at once, with the measured neighbour offsets as strong constraints and the robot positions as weak anchors, clamped near the robot.
- **Exposure equalisation.** One gain per photo and colour channel, solved by least squares on the overlaps.
- **Blending.** Each pixel is a weighted average of the photos covering it, with weights that fall off steeply away from each photo's centre, where lens distortion is lowest.
- **Levelling.** A slightly rotated camera makes the tray slope; the panorama is rotated back by the measured angle.
- **Model layout.** Each tray row is cut out and turned into a vertical column about 140 px wide, the layout of the photos the models were trained on. A copy 4× larger (560 px columns) is kept for the lithology model, whose tiles would otherwise be too small.

One recorded 120-photo scan took about 17 seconds and produced a 12.7 px/cm model image. These are example measurements, not fixed output settings or a hardware-independent benchmark.

### 2. Known grid

When every filename carries a grid position such as `capture_r2c4_001.png`, rows are built left to right, then stacked. Each overlap is measured by normalised template matching against the previous tile. Grid and blind outputs are cropped panoramas; they do not receive the robot branch's conversion into vertical model columns.

### 3. Blind matching

For photographs without usable robot or grid positions, the code tries OpenCV Stitcher in `SCANS` mode, then SIFT matching with a RANSAC homography. Repetitive or low-texture trays can produce incorrect matches or omitted frames; inspect the assembled image before analysis.

## Robot manifest

The stitcher looks for `scan_manifest.json` in the image folder, then up to two parent directories. It reads:

- `grid.definition.primary_axis`: `"x"` or `"y"` (defaults to `"x"`);
- `passes.camera.points`: a list of capture records;
- each record's `filepath`, `row_index`, `sample_index`, `x` and `y`, with coordinates in millimetres;
- optional `status`, which must be `"ok"` if supplied. `row` and `sample` are accepted aliases for the index fields.

File basenames must match the loaded images. At least two matched records with equal image shapes are required, and assembly must successfully estimate the scale. Otherwise the code falls through to the grid or blind strategy.

## Scale and measurement

The robot branch estimates scale from robot displacement and adjusts it for the model layout. Grid and blind stitching look for a **5 cm ArUco marker from `DICT_4X4_50`**. If none is detected, or marker detection fails, the returned value is **50 px/cm**.

That fallback is a default, not a measured calibration. Verify it against a known physical distance before passing it to `analyze`. Any later image resizing must also update the scale. The output should have vertical core columns in left-to-right reading order; the pipeline warns when detected column widths fall outside its training range of 50–260 pixels.
