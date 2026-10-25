# SuperVario

Windows desktop and command-line application for building **reviewable variogram
candidates** for Leapfrog Geo / Edge. Processing stays local and input files are
read-only.

## Start

The project environment is `.venv`. Double-click `run_gui.bat`, or run:

```powershell
.\.venv\Scripts\python.exe -m supervario gui
```

To install on another computer with Python 3.11 or newer:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The application currently relies on local table-reading/filtering modules and
the sibling projects `MeshOps` and `Anterpolator`. It is **not a standalone executable**.
Keep their directory layout, or set `SUPERVARIO_PROJECTS` to their common parent
folder. Their source files and environments are not modified.

## Desktop workflow

Input widths follow their content: file paths grow with the window, column/sheet
names have medium-width controls, counts stay compact, and distances/values have
more room. Labels sit directly beside their fields; unused space stays at the
end of grouped rows rather than stretching every control equally.
The supplied program artwork is stored in `supervario/assets/supervario.png`,
with a multi-resolution Windows icon in `supervario/assets/supervario.ico`.
The GUI uses it for application/window icons and sets a dedicated Windows
taskbar identity. The batch launcher itself remains a Windows command file;
Windows does not display a custom icon directly on `.bat` files.

Hover over the sample/pair caps, structure count, search increments, manual
orientation controls or contact settings (and their adjacent labels) for brief
explanations of their purpose and limits.

1. Choose a sample CSV or Excel workbook, the worksheet if needed, X/Y/Z and the
   numeric value column. **Read columns / detect mappings** detects common
   coordinate names and populates editable selectors.
   Optionally map **Hole ID**, **From** and **To** (recommended) from the sample
   file for spacing-informed automatic lag distance. Hole ID alone is supported;
   From/To require Hole ID and must be mapped together. Blank mappings disable
   this feature. Coordinates must already be desurveyed sample midpoints;
   interval depths must use the same linear units. This does not perform
   desurveying or compositing. Mappings are saved as `samples.hole_id`,
   `samples.from` and `samples.to` in project JSON. Retained mapped rows must
   have nonblank hole IDs and finite `0 <= From < To`; invalid inputs are rejected.
2. Select either a flagged block model or named solid files:
   - Blocks: choose the domain field, centroid coordinates and per-row DX/DY/DZ.
     An explicit regular `block_size` can be set in a saved project JSON file.
   - Solids: add one file per domain; edit the domain names in the table.
     Formats include DXF triangulations, Leapfrog MSH, supported Vulcan 00t
     variants, and trimesh formats such as STL/OBJ/PLY.
3. Edit filters using the filter dialog. It supports
   categorical include/exclude, numeric bounds, present values, expressions and
   explicit blank handling. `BM:<field>` means the attribute of the block
   containing the sample. Filters are AND-combined and applied after flagging.
   Previews of block values come from the source block field; the actual filter
   is evaluated on the flagged samples during calculation.
4. Set lag distance, bin count, distance, cone half-angle, minimum pairs, sample/pair
   budgets and one or two spherical structures. Zero lag/distance selects the
   automatic defaults (the GUI displays **Automatic**, rather than a numeric zero,
   in these two distance fields). The configured lag bin count must be at least 3
   (default 12). Spacing-informed automatic selection may adjust the effective
   bin count to preserve maximum distance; the review summary shows the actual
   lag distance, bin count and selection method. Sample cap must be at least 12, pair cap and minimum pairs
   at least 1, and the cone half-angle greater than 0. The GUI enforces these
   minima; invalid project/CLI settings are rejected before reading input files.
   Supply a manual dip azimuth, dip and pitch, or use the
   coarse automatic direction search.
5. Set the automatic azimuth, plunge and in-plane search increments as needed.
   String Theory orientation import is deprecated; no external tendencies file
   is required.
6. Calculate. Processing runs in a worker thread; inspect the run log for rejected
   rows and failed domains. Cancellation is cooperative at the next progress
   checkpoint, not an interruption of a native read/repair/fit call. The window
   must remain open until calculation or cancellation finishes.
   The log is a resizable right-hand panel, not a tab. **Hide log / Show log** or
   its close button collapses it without losing messages; it remains accessible
   during calculation. The bottom status shows the latest processing step and
   elapsed time, with a busy indicator (not an overall percentage). CSV reading
   reports approximate per-file progress; native repair/fitting calls may take
   time between step updates.
7. Review the radial plot and three axis-aligned experimental/model plots.
   **Major**, **Semi-major** and **Minor** buttons above the axis charts toggle
   each variogram independently. Visible variograms share the available height;
   all three start visible. This changes only the display, not the model or
   experimental values. The radial plot stays visible.
   The review summary shows the calculated **Lag distance**, including when
   automatic selection was used; standalone LFVs do not contain this setting.
   Marker size reflects pair count. Edit nugget, contributions, ranges or spherical/linear
   types and apply the changes. This invalidates the automatic fit statistics.
   Changing orientation requires recalculation, not just relabelling existing
   experimental curves.
   **Structure directions** optionally overlays projected axes: red major, green
   semi-major, blue minor; solid S1 and dashed S2 when orientations differ.
   Shared orientations use one set labelled **S1/S2**, not duplicate arrows.
   Bidirectional arrows indicate
   orientation, not fitted ranges; different display lengths distinguish structures.
   The plot remains in the experimental major/semi-major plane. Perpendicular
   axes are listed as **Out of plane**, not drawn as misleading radial arrows.
   Newly calculated reports retain the plane rotation explicitly; older reports
   use the first structure's rotation as the display-plane reference.
   Compare colours at the **same distance**: lower semivariance generally means
   stronger continuity. Blank cells have no pairs, not zero variability. The
   minor direction cannot be assessed from this plane alone. Axis names identify
   the fitted directions. New automatic fits enforce
   **major >= semi-major >= minor** for every structure. Older/imported LFVs
   retain their original parameters and are not silently rewritten.
   **Radial colour scale** offers Rainbow (default), Turbo, Viridis and Cividis.
   Changing the palette only changes the display, not the model or experimental
   values. A hint below the charts explains how to compare continuity at equal
   distances, missing pairs and the limitations of this two-dimensional view.
   Plot-reading guidance and fit warnings sit beside the kriging recommendations
   in independently scrollable panels. Drag the horizontal divider above these
   panels down to enlarge the charts (or collapse the text), and back up to
   restore them. Drag the divider between the text panels to adjust their widths.
   Radial angles follow a **Leapfrog-style pitch display, not compass bearings**:
   zero is at the right, 90 degrees at the bottom, increasing clockwise from the
   plane's strike reference. The major arrow appears at its fitted pitch rather
   than always at zero. Colours and arrows are rotated together; the experimental
   values and fitted models are unchanged. Plane Dip/Dip Azimuth are in the title.
   A horizontal red arrow does not mean east-west.
   The model summary lists actual compass azimuth/downward plunge
   for each axis of the first/reference structure. Plane dip azimuth is also not
   the major-axis azimuth; pitch determines its direction within the plane.
   Axis variogram charts start both axes at zero. A labelled marker at the
   y-axis shows the nugget (normalised when normalised display is selected).
   Model curves start at the positive-distance limit, the nugget, rather than
   drawing a false ramp from the exact-zero semivariance. This is display-only;
   model evaluation still has zero semivariance at exactly zero separation.
   **Downhole diagnostic...** opens a separate short-range review when Hole ID,
   From and To are mapped. It uses same-hole interval-midpoint depth differences,
   not XYZ distances, with median interval length as lag and 12 centred windows
   `[(k+0.5)*lag, (k+1.5)*lag)` for k=0..11. Zero-depth pairs are excluded.
   Points show pair and contributing-hole counts; grey points fail `min_pairs`
   or have fewer than two holes. Pair sampling is reproducible and bounded by
   the pair budget, using the fitting sample cap and any approved borrowed samples.
   There is no directional fitted curve: holes can have different orientations.
   The current nugget and sill are reference lines, not a downhole fit.
   When drilling evidence is sufficient, calculation automatically extrapolates
   the first three contiguous depth bins to zero and refits the directional
   model with that nugget fixed. This is a conservative heuristic, not a measured
   zero-distance nugget. It requires at least `max(30, min_pairs)` pairs per bin,
   three common holes contributing at least three pairs per early bin, and no
   hole supplying over 50% of an early bin's pairs. Overlapping intervals or
   interval-length ratios above 2:1 disable automatic estimation.
   A pair-weighted straight-line fit must have nonnegative slope and intercept,
   intercept no higher than the first three bin values, and residual RMSE no
   greater than 5% of data variance. Available four/five-bin windows and
   leave-one-hole-out intercepts must stay nonnegative and within 10% of variance
   of the three-bin estimate. These sensitivity bounds are not confidence intervals.
   A converged fixed-nugget refit must not worsen directional normalised RMSE by
   more than 0.02. Otherwise the original fit remains, with the rejection reason
   logged and saved. The downhole window displays the automatic status,
   extrapolation and sensitivity range, while retaining manual review/refit.
   `downhole.automatic_nugget` records the decision and checks; accepted refits
   preserve the original model/fit under `fit.automatic_downhole_refit`.
   Orientation is retained, not revalidated; geological review is still required.
   The first point is not a measured zero-distance nugget. Mixed support,
   interval overlaps and trends require review; no compositing or overlap
   correction is performed. Diagnostics and limitations are exported under
   `downhole` in `analysis.json`.
   **Normalised values** switches both the reviewed nugget input and Y-axis
   between absolute semivariance and semivariance divided by data variance.
   A labelled marker at the Y-axis and a horizontal nugget line update while
   editing, together with the preview total sill. Switching units preserves
   the underlying absolute nugget; previews do not change the model.
   To use this evidence, enter a **Reviewed nugget** in the selected units and explicitly
   choose **Apply reviewed nugget**. Contributions, ranges and orientations
   stay unchanged; total sill is recomputed, fit statistics are invalidated,
   recommendations refresh and the previous/new nugget is recorded in
   `fit.downhole_nugget_reviews`. Review the resulting model before export.
   Close the dialog without applying to keep the model unchanged.
   Alternatively, **Refit with reviewed nugget fixed** optimises spherical
   contributions and ordered ranges against the saved directional experimental
   data, keeping the entered nugget and shared orientation fixed. Existing
   minimum-pair thresholds, global pair weights and supported-distance range caps
   still apply. Total sill is free, not forced to the variance or downhole plateau.
   Fit statistics, plots and recommendations refresh; the previous model/fit and
   reviewed nugget are retained in the review history. Orientation is not
   reassessed and unresolved-direction warnings remain. This requires newly
   calculated results with saved fitting settings; standalone LFVs cannot be refitted.
8. Review contact profiles. Suggestions are **not approvals**. Check the domain
   pairs you approve and explicitly recalculate to share samples. Unchecked pairs
   remain hard boundaries.
9. Export reviewed LFVs, `analysis.json` and `estimator_recommendations.json`. The report contains configuration,
   sample/pair counts, experimental lag values/counts, contact diagnostics,
   approvals, model parameters, manual-edit markers, warnings and domain failures.
   Export uses the calculation snapshot, not any subsequently edited inputs.
   Existing outputs require explicit replacement approval.

### Estimator and search-pass recommendations

Context-sensitive **Pass selection and additional-pass guidance** explains the
reference structure, base search semi-axes, each retained pass's multiplier and
support gain, and why no further pass is justified by the current rule. It flags
unmet support goals, extrapolative/conditional searches and unresolved directions.
This guidance appears in the review panel, JSON `pass_guidance` and each workbook
domain tab. It explains the existing selection; it does not add optional passes
after the support goal is reached or establish actual target-block coverage.

Each successful domain includes **ordinary kriging starting settings**, with
IDW (exponent 2, test 1/2/3) and nearest-neighbour comparison guidance. View them
in the scrollable recommendations panel below the variogram plots, or open
`estimator_recommendations.xlsx` for a human-readable workbook, or
`estimator_recommendations.json` for structured data. Both are exported by the
GUI and CLI. The workbook contains one formatted tab per domain and a filterable
**Summary** table with one row per domain/pass, including Leapfrog ranges,
directions, sample limits, adoption status and warnings. Domains without
supported passes still appear in the summary with their unresolved status.
Domain tabs include the detailed Leapfrog settings, validation guidance,
unspecified parameters and warnings. Tab names are sanitised/truncated to Excel's
31-character limit, with collision suffixes; full domain names remain in cells.
The workbook has frozen headers, wrapped text and formatted numeric values.
Recommendations are advisory outputs, not fields imported
by the LFV file and not a fitted/validated estimation strategy.

Recommendations use the tabs and field names in the
[Leapfrog Edge kriging estimator](https://help.seequent.com/Geo/5.1/en-GB/Content/estimation/estimators.htm):

- **Interpolant:** Kriging type = Ordinary; Mean value is not applicable.
  Simple kriging requires a justified stationary domain mean, not an automatically
  substituted sample mean. Discretisation X/Y/Z remains unspecified for blocks
  until block dimensions and convergence are checked; 1/1/1 is for point kriging.
- **Ellipsoid**, per pass: Maximum = major, Intermed. = semi-major,
  Minimum = minor, plus Dip, Dip Azimuth and Pitch from the reference structure.
  These are centre-to-boundary distances, not diameters. Preserve axis identity:
  do not sort imported ranges independently of orientation. Variable Orientation
  is off. The search ellipsoid is distinct from the variogram ellipsoid; each
  pass has its own scaled search ranges.
- **Search:** Minimum samples / Maximum samples = 8/24 as starting candidates,
  not optimised values or Leapfrog defaults. Outlier Restriction is off pending
  selection of Clamp/Discard, Distance (% of search) and Value threshold from
  domain review. Sector Search is off for the baseline, with Octant (3 samples
  per sector, 4 empty sectors) proposed only as a sensitivity test. Drillhole
  Limit, Maximum samples per drillhole and Apply drillhole limit per sector need
  drillhole/composite evidence and validation; none is inferred from XYZ alone.
- **Value Clipping:** Clip input values is off; Lower/Upper bounds are not
  selected automatically. This is separate from distant-outlier restrictions.
- **Outputs:** Value and Status are always included by Leapfrog. Select NS,
  MinD, AvgD, ND, KM, KV, KE, SoR, Sum and SumN for validation. These outputs
  must be calculated in Leapfrog; SuperVario does not simulate a kriging run.

The GUI text follows these sections and the JSON report includes a structured
`leapfrog_parameters` entry with per-pass Ellipsoid/Search fields. This is
manual-entry guidance, not an importable estimator file. Screenshot example
values (Simple mean 100, discretisation 5/5/2, sample limits 4/20) are not
automatically adopted as recommendations. Pass counts, support calculations,
orientation warnings and conditional-expansion safeguards remain unchanged.

The tool evaluates five candidate search ellipsoids at **0.5, 0.75, 1, 1.5 and 2**
times a reference structure's ranges. It uses the positive-contribution spherical
structure with the largest geometric-mean range, retaining that structure's own
rotation rather than combining incompatible orientations. Each base semi-axis
is capped at the observed maximum pair distance; extrapolated fitted ranges
and later expansions are flagged. Non-spherical models do not receive inferred
search dimensions. Nugget-dominated models carry an additional caution.

The suggested sample minimum/maximum is **8/24**, a conservative starting policy,
not a data-derived optimum. Raw neighbour counts are measured at up to 1,000
deterministically selected unique hard-domain sample locations, against up to
5,000 unique source locations (including explicitly approved borrowed samples).
Zero-distance locations are excluded, so duplicate coordinates cannot inflate
support. When sources are capped, counts are lower bounds. Sector and drillhole
constraints are **not** applied to these support fractions.

The first candidate is retained. Additional candidates are retained only when
they add at least one percentage point of supported sample probes, stopping once
90% of probes have eight other unique neighbours. This usually produces **1-5
starting passes**, not a mandatory five. All five default diagnostics are saved.
If support is still insufficient, the tool also tests **3x and 4x** ranges,
retaining them only when they improve support by the same one-percentage-point
criterion. This permits **more than five passes (up to seven)** in sparse contexts.
These extra candidates are explicitly **conditional/exploratory**, not approved
production settings; their use requires target-block evidence and geological
approval. Default and conditional pass counts are reported separately. No
indefinite expansion or forced coverage is recommended, and no boundary policy
is relaxed by a larger search.

Each pass reports major/semi-major/minor **semi-axis lengths** (centre to boundary),
full axis lengths for clarity, dip azimuth/dip/pitch, sample limits and measured
sample-probe support. Later passes should estimate only blocks not estimated by
earlier passes; preserve pass identity and do not equate it to resource classification.

Octant search with maximum 3 samples/sector and maximum 4 empty sectors is offered
only as a sensitivity-test candidate. Drillhole limits and block discretisation
are deliberately left unset: XYZ and a variogram alone cannot justify them.
The validation checklist includes sector/hole occupancy, spatial or drillhole-held-out
validation, actual target-grid unestimated volume, block-discretisation convergence,
swath plots, kriging variance/efficiency/negative weights and OK/IDW/NN bias checks.
Clipping and outlier thresholds are not inferred automatically.

**This does not run kriging, optimise a neighbourhood or measure target-block
coverage.** Sample-location support can be optimistic relative to unsampled
blocks, particularly with clustered drilling. It does not prove independent
drillhole support. Recommendations refresh after manual model edits and again
on export, using the saved coordinate evidence. The analysis report therefore
contains those bounded coordinate subsets in addition to the configuration.

**Open LFV** opens a reference model separately for inspection, editing and Save As.
It never fabricates experimental curves from the model. The supplied reference is
in [Reference](Reference).

Less common options such as `bandwidth`, `seed`, `block_size`, `max_voxels`,
and stored `approved_pairs` remain available in saved project JSON files.
Use **Save project**, edit the file in a text editor, then **Load project**.
Settings without dedicated form controls are preserved when saving again.
Filters can also be edited this way when a BMF cannot supply a GUI value preview.
There is no embedded JSON editor tab.

**Save project** can update an existing project JSON after saving or reloading it.
Sample, block, solid and tendency source files remain protected from overwrite.
Analysis/LFV exports also protect the loaded and saved project files.

## File handling and domain assignment

- CSV reading supports separator, decimal-mark, encoding and
  Leapfrog metadata handling. Blank format fields select detection. A header-row
  override is **1-based**; `0` means automatic first non-comment header. Arbitrary
  prose before the header needs an explicit row. Quoted headers are supported.
- `.xlsx` and `.xlsm` use openpyxl; Excel is not required. These workbooks are
  loaded into memory. Legacy binary `.xls` is not supported.
- CSV block models are streamed in 100,000-row chunks. Samples read only needed
  columns, but the retained sample table and contact calculations are in memory.
  BMF inputs currently use Anterpolator's in-memory reader, not the streaming CSV
  path. Multi-gigabyte BMF/Excel inputs should be converted to CSV.
- Axis-aligned blocks are matched by **actual extents**, not nearest centroids,
  so mixed-size subblocks are supported. No sizes are guessed from centroid
  spacing. Subdivision metadata alone is not enough for mixed-size subblocks:
  provide per-row dimensions. Rotated block models must first be transformed
  into an axis-aligned coordinate system.
- Block boxes are half-open: the minimum face is included, maximum face excluded.
  Shared faces belong to the block on the positive side. Samples exactly on the
  outer maximum face are outside. Overlapping blocks containing the same sample
  stop the calculation rather than silently choosing a domain.
- All sample and domain coordinates must use the same Cartesian CRS and linear
  units. There is no implicit reprojection or coordinate-unit conversion.
- BMF is **variant-dependent**. The adapter uses Anterpolator's available readers,
  rejects undecodable/incomplete tables and recommends an explicit flagged CSV
  export. There is no claim of universal or licensed Maptek SDK support.
- 00t likewise depends on variants handled by MeshOps. Unsupported variants fail
  explicitly. DXF requires supported triangulated face/polyface geometry.

### Mesh repair and coherent voxelization

`none` requires a closed, consistently oriented positive-volume mesh. A global
inward winding is reversed on import without moving geometry, and is logged.
This is a basic validity gate, **not** a self-intersection certificate.
Exact containment rejects domain overlap detected at sample locations; it does
not certify that whole solids are disjoint. Surface-coincident points have
numerically ambiguous containment; review them outside this application.

`meshops` calls the existing MeshOps solid-healing pipeline, with residual alpha-wrap
fallback disabled, then requires a valid closed volume. Some repairs require
optional MeshOps dependencies such as MeshFix/manifold/VTK; dependency failures
are shown, never converted into successful containment. For complex repairs,
use MeshOps directly and inspect the repaired solid before using it here.

`voxel` is an **explicitly approximate** raster-fill repair:

- Every domain uses the same world-aligned lattice and pitch.
- Voxel ownership is exclusive. Table order is priority: **first solid wins** a
  conflicting cell. The report records pitch, priority and conflict count.
- Even disjoint continuous solids can touch the same coarse voxel; conflict
  resolution can erode narrow domains and move boundaries by voxel-scale distances.
  This guarantees exclusive *voxel ownership*, not exact recovery of source solids.
- A bounding-grid/occupancy budget prevents excessively fine allocations. The
  default is 2,000,000 cells; increase pitch when rejected.
- Raster filling is not guaranteed to reconstruct severely damaged/open surfaces.
  This mode assigns samples from an in-memory ownership grid; it does not export
  repaired domain meshes.

## Scientific conventions and limits

- Classical Matheron estimator: `gamma(h) = mean((v_i - v_j)^2) / 2`, with unordered
  sample pairs. Coincident pairs are excluded and their drawn count is reported.
- Population variance (`ddof=0`) is computed over the full selected domain sample
  set, including approved borrowed samples, before optional sample reduction.
- There is **no implicit compositing, declustering, transformation, capping,
  detrending, robust estimator or downhole variogram**. Prepare comparable sample
  support and investigate stationarity before modelling.
- Default computational budgets are 5,000 samples and 300,000 drawn unordered
  pairs per domain, with a reproducible seed. Sample reduction and pair sampling
  are uniform without replacement and explicitly reported. Pair sampling occurs
  before distance/cone selection; short-lag support can therefore be sparse.
- Default maximum distance is half the selected sample bounding-box diagonal.
  Without drilling mappings, default lag distance is maximum distance / bin count.
  With Hole ID mapped and lag set to Automatic, each domain uses the median
  positive nearest different-hole sample distance to propose 0.5x, 1x and 2x
  spacing candidates. The median From/To interval length, when mapped, is a
  lower bound on candidate lag distance. Bin counts are adjusted (3-1000) to
  preserve the maximum distance. The finest candidate is retained when at
  least half of its pooled bins and three bins on each screening axis meet
  `min_pairs`. Screening axes use the manual rotation when supplied, otherwise
  principal coordinate axes; final automatic directions can differ.
  This is a transparent spacing/support heuristic, not a unique statistical
  optimum. It does not measure independent hole-pair support or optimise
  downhole/short-range resolution. One lag distance is used for all directions.
  Fewer than two holes, no positive between-hole distances or insufficient
  candidate support retains the geometric default with an explicit log/warning.
  Diagnostics, spacing/interval quantiles, candidate support and selection
  reasons are saved as `lag_selection` in `analysis.json`. They use the same
  deterministic sample cap as fitting (and approved borrowed samples, if any).
  An explicit nonzero lag distance always overrides automatic selection.
  For manual lag distance, the effective maximum is
  `min(max_distance, lag * bins)`. Lag bins are `[k*lag, (k+1)*lag)` and plotted at
  mean pair distance; empty bins are JSON `null`, not zero semivariance.
- Directions use bidirectional cones in 3D and optional perpendicular bandwidth
  in coordinate units. At least three bins per axis must meet `min_pairs`.
  Sparse bins remain visible but do not influence the fit.
- Automatic orientation screens an axial grid at configurable azimuth and plunge
  increments (defaults **15 degrees azimuth / 30 degrees plunge**), then screens
  perpendicular directions. GUI controls **Azimuth search step** and **Plunge
  search step** correspond to `variogram.azimuth_step` and
  `variogram.plunge_step` in project JSON. Both accept multiples of 5 from 5 to 90
  degrees; missing settings preserve the defaults. Plunges are symmetric multiples
  of the chosen increment about horizontal, excluding the poles, plus one
  bidirectional vertical candidate. Azimuth covers [0, 180); perpendicular
  screening also uses the azimuth increment. Finer grids cost more computation
  and do not create experimental information. It
  compares semivariance over the **same early lag bins** for every candidate.
  The common window is the first `max(2, ceil(bins/3))` bins supported by at least
  half of candidates with three supported bins. Candidates must support every bin
  of that window and at least three bins overall. The cone bandwidth, if set,
  also applies to direction search.
  The score is mean normalised semivariance across common bins, plus penalties
  `0.5*(1-distance_coverage) + 0.25*(1-bin_occupancy) +
  0.25*(1-sqrt(relative_pair_count))`. Coverage is the last supported bin relative
  to the longest candidate extent, occupancy is supported-bin count relative to
  that extent, and relative pair count compares all cone pairs to the best
  populated candidate. These are explicit heuristic coefficients, not a
  calibrated uncertainty model. They reduce the attraction of low-semivariance
  directions with very short or sparse support.
  Four deterministic half-pair subsamples repeat this initial ranking against the
  same comparison window. The final fitted major direction is assessed after axis
  reassignment: at least three of four trial winners must lie within 30 degrees,
  it must cover at least half the final directional distance extent, and support
  at least `max(3, ceil(bins/3))` bins. A competing candidate more than 30 degrees
  away with score within 0.1 also makes the direction ambiguous. When the
  joint frame comparison below selects an alternative frame, its frame-based
  sensitivity ranking replaces this early-lag-only assessment: four half
  held-out-pair subsets re-rank the fixed training fits, with a 0.01 score
  ambiguity margin. Each repeat uses common bins with half the held-out
  minimum-pair threshold (rounded up), retaining at least three and half the
  original comparison bins on every axis; otherwise that repeat is unresolved.
  These are not four independent refits.
  For an alternative selected frame, both the major direction **and plane normal**
  must have three of four repeat winners within 30 degrees. Competing eligible
  frames differing by more than 30 degrees in either major direction or normal
  are checked for score ambiguity. Retaining the baseline retains its original
  direction assessment rather than treating a lone eligible frame as validation.
  These are pair-resampling sensitivity checks, **not independent sample or
  drillhole resampling, confidence intervals, or geological validation**.
  Full candidate rankings, common bins, trial directions, angular differences
  and resolution status are saved in `orientation_diagnostics`. The GUI displays
  **major_direction_unresolved** when the checks fail; an exploratory model may
  still be inspected/exported, but its estimator recommendations explicitly
  warn against production use. If no comparable candidate exists, the domain
  fails explicitly rather than inventing a direction. Manual orientations remain
  user-specified and are labelled not statistically validated.
- Nonnegative nugget/contributions and positive ranges are fitted jointly across
  all three axes using bounded nonlinear least squares and three starts.
  Ranges are parameterised to enforce major >= semi-major >= minor for **both**
  nested structures throughout optimisation, not sorted independently afterwards.
  Each range is capped at the farthest supported lag-bin mean distance on that
  axis (also no greater than the configured maximum). A cap reached means
  unresolved continuity, not proof that the true range ends there.
  Automatic mode fits all six assignments of the searched orthogonal directions
  to named axes and keeps the lowest weighted residual. Rotation, directional
  experiments, radial plane and estimator recommendations follow the winning
  assignment together. Manual rotation is preserved, with ordered fitting along
  the user-specified axes; a contradictory direction may therefore fit less well.
  Both structures share the fitted orientation; differing orientations are not
  estimated automatically.
  The initial frame no longer fixes the only plane that can be selected.
  Automatic mode also shortlists **complete 3D orthogonal frames with alternative
  plane normals**: up to four promising major seeds, separated by at least
  25 degrees, each with up to two similarly separated perpendicular directions.
  A triad must have comparable supported directional scores on all three axes.
  The best four geometrically distinct triads by summed directional scores are
  retained; each contributes all six named-axis assignments (at most 24 additional
  ordered frames before duplicate removal). This bounded screening is a heuristic
  shortlist, not exhaustive optimisation over all frames.
  After the six-axis assignment, automatic mode compares **complete orthogonal
  frames rotated within the fitted major/semi-major plane**, at configurable
  increments, with both in-plane axis assignments. **In-plane search step**
  defaults to 15 degrees; values from 5 to 90 in multiples of 5 are accepted.
  Set `variogram.in_plane_step` in CLI project JSON; projects without the setting
  retain the 15-degree default. A 5-degree step tests a finer grid but costs more
  computation and does not create additional experimental support. This setting
  does not change radial display resolution, cone half-angle or the separate 3D
  search increments. The chosen step is recorded in comparison diagnostics.
  In-plane proposals keep the minor direction and plane unchanged; shortlisted
  3D proposals can change both. All proposals compete in the **same joint
  comparison**, rather than selecting each plane in isolation.
  Each frame is jointly fitted, with major >= semi-major >=
  minor still enforced. This allows a compromise with higher short-lag
  semivariance but longer supported structured continuity.
  A deterministic two-thirds/one-third pair split provides training fits and
  held-out prediction errors. The baseline defines fixed validation lag indices
  on each axis (at least three per axis). Alternatives lacking support in any
  of these bins are rejected rather than shortening everyone's comparison window.
  The full-data model is retained for output;
  the separate training fit supplies the held-out prediction. Shared samples and
  the initial full-data orientation search mean this is a **sensitivity check,
  not independent predictive validation**.
  The range measure is the sum, over structures contributing at least 10% of
  total sill, of `contribution / sill * sqrt(major_range * semi_major_range)`.
  A structure earns range credit only if **both** in-plane curves have at least
  three supported bins below their range and two at or beyond 1.1 times their
  range, in both full data and common held-out bins. Thus a fitted range at the
  distance cap, a negligible structure, or extra distant pairs alone cannot
  manufacture a longer continuity reward.
  A replacement needs at least a 5% increase in this supported range measure
  (with the measure also at least one lag width), at least 80% of baseline validation pair support,
  and held-out normalised RMSE no more than 0.02 worse than baseline.
  Eligible frames minimise `held_out_RMSE + 0.1*(1 - supported_range / R)`,
  where `R` is the larger of one lag width and the largest compared supported
  range measure. A longer range therefore earns at most 0.1 normalised-error
  units of preference; it does not override the error/support guards.
  These thresholds are transparent heuristics, not a geological optimum.
  Insufficient comparable evidence retains the original frame with an explicit
  diagnostic. The GUI shows the plane-comparison outcome; the report saves every
  candidate source, plane normal, rejection reasons, range measure, scores and sensitivity winners in
  `fit.plane_refinement` and `orientation_diagnostics.plane_refinement`.
  The 3D shortlist policy/counts and configured grid increments are saved in
  `orientation_diagnostics`. Selection can report `alternative_3d_frame_selected`,
  `rotated_frame_selected`, or `baseline_retained`, with explicit statuses when
  comparable evidence is insufficient. This finite shortlisted search is
  **not a proven global 3D optimum**.
  Manual orientations bypass all automatic frame searches. Experiments, radial plot and estimator
  ellipsoids all use the final selected frame.
  Weights are proportional to square-root pair counts with **one global
  normalisation across all supported bins and directions**. A sparse axis no
  longer gets the same total weight as a well-supported axis. Total
  sill is free; it is **not** forced to the data variance. Reported normalised
  weighted RMSE uses those global pair weights. The nugget remains jointly
  fitted; nearest supported lag distances are reported and the lack of an
  independent zero-distance nugget measurement is stated explicitly.
  Fits hitting bounds
  or reaching supported-distance limits are flagged. Automatic fits no longer
  extrapolate ranges to four times the observed distance. Boundary solutions,
  short-range resolution limits and sampling budgets still require review:
  constraints cannot manufacture missing information. Check the reported residual
  and experimental curves, particularly if two structures prefer crossing
  anisotropies that the shared ordered-axis model cannot represent.
- Automatic models use one or two spherical structures. Manual linear curves
  are unbounded. LFV spheroidal structures can be validated and round-tripped,
  but their proprietary curve/alpha semantics are not approximated or fitted.

### Boundary review

Contact profiles use distance to the nearest sample in the opposite domain,
**not signed distance to the actual solid or block-domain surface**. Suggestions
require the configured sample count on both sides (default 20), a standardised
mean difference at most 0.25 and a two-sample KS statistic at most 0.2. These
descriptive thresholds are not an equivalence test, do not correct for spatial
dependence, and do not establish a geological soft boundary.

No samples are shared automatically. An approved pair permits filtered samples
from that neighbour only when they are within the configured distance of a
target-domain sample. Sharing is symmetric by approved pair, direct only (not
transitive), and uses original domain assignments. Geological approval must take
account of this sample-distance approximation.

### String Theory orientation import (deprecated)

Anterpolator's String Theory connects nearby similar-valued samples using
grade-difference and distance thresholds, builds discrete grid paths, filters
path orientations by frequency, and fills intermediate cells. Its statistics
describe these selected paths. This is an interpolation/path-analysis method,
not an independent measurement of variogram continuity. Connection frequencies
depend on sampling geometry, grade thresholds, grid resolution and filtering;
path lengths are not structured variogram ranges.

SuperVario previously used only a small orientation histogram from the
`path_stats_*.csv` export to seed its search. That narrow role overlaps the native
sample-pair direction screening and complete 3D-frame comparison now available.
There is no demonstrated additional benefit here to justify implementing a
second interpolation engine, its parameters, generated point clouds and plots.
The algorithms are not identical and the assessment does not establish that
String Theory is redundant for interpolation in Anterpolator.

The import control has therefore been removed and replaced by a deprecation
notice. Legacy `variogram.trend_file` values are retained for provenance but
**ignored without opening the file**, in both automatic and manual modes.
Configured legacy inputs produce an explicit run-log message, a fit warning and
`string_theory.status = deprecated_ignored` in the report; GUI-loaded legacy
projects also show a notice. Missing legacy statistics/point-cloud files no
longer cause domain failures. Their paths remain protected from output
overwrites. The legacy parser remains available for inspecting old statistics,
but is no longer called by fitting.

Use the native 3D search diagnostics and radial/directional variograms for this
workflow. Interpolated point clouds must not be fed back as additional measured
samples: doing so would manufacture pair support and bias the fitted variogram.

## CLI

```powershell
.\.venv\Scripts\python.exe -m supervario init project.json
.\.venv\Scripts\python.exe -m supervario header samples.csv
.\.venv\Scripts\python.exe -m supervario header samples.csv --sep ";" --decimal "," --header-line 3
.\.venv\Scripts\python.exe -m supervario run project.json
.\.venv\Scripts\python.exe -m supervario run project.json --overwrite
.\.venv\Scripts\python.exe -m supervario inspect-lfv "Reference\Variogram Model RQDpc UGB01.lfv"
.\.venv\Scripts\python.exe -m supervario repair-mesh original.00t repaired.msh
.\.venv\Scripts\python.exe -m supervario demo examples\my_demo
.\.venv\Scripts\python.exe -m supervario gui examples\my_demo\project.json
```

Project input/output paths are resolved relative to the project JSON. Use absolute
paths when editing unsaved GUI JSON. The demo command refuses to replace input
files. `repair-mesh` refuses to overwrite existing destinations.

CLI status: `0` success, `1` fatal error/no successful models, `2` partial domain
failure (successful LFVs and a report are still exported). Failed domains never
receive an empty or fabricated LFV. Outputs use readable names such as
`variogram_UGB01.lfv`, sanitising unsupported characters and shortening long
domain names. Only collisions within an export receive numeric suffixes
(`_2`, `_3`, etc.), including Windows case-insensitive collisions. Natural domain
filenames are reserved before assigning suffixes. Existing files still require
explicit overwrite approval; they do not cause a new suffixed export.

## LFV compatibility and verification

LFV output follows the supplied reference JSON `content_type`
`com.seequent.elements.variogram`, version `2`, including absolute nugget/sills,
data variance, domain/attribute metadata, rotation and nested anisotropic ranges.
Normalised display divides by data variance, not total sill. The reference
round-trips without changing its contents, and its rotation reproduces the
approximate 57-degree plunge / 116-degree azimuth shown in the supplied screenshot.

**Import acceptance has not been tested in a running licensed Leapfrog Edge
installation.** Confirm orientation, units and model behaviour there before using
results for estimation. This application is not a substitute for geological review.

Run validation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Tests cover known spherical/nested numerical curves, directional bins/rotation,
pair budgets, reference LFV round-trip, CSV/Excel formats, mixed-size blocks,
overlap rejection, actual MSH/DXF/STL ingestion, shared voxel ownership, filters,
explicit BMF failure, soft-boundary opt-in, exports/provenance, legacy String Theory
deprecation/provenance, and offscreen GUI plotting/editing/background execution. Real proprietary
BMF/00t variants and multi-gigabyte datasets need representative project validation.

See also Seequent's
[Edge overview](https://help.seequent.com/Geo/5.1/en-GB/Content/estimation/estimation.htm)
and [variography documentation](https://help.seequent.com/Geo/5.1/en-GB/Content/estimation/variograms.htm).
