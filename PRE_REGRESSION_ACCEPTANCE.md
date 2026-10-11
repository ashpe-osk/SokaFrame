# Pre-regression acceptance gate

Run from the repository root with the CV environment:

```powershell
& "C:\omniBoard-Sportz\.venv-1\Scripts\python.exe" `
  "C:\omniBoard-Sportz\scripts\run_pre_regression_checks.py"
```

The gate checks Python syntax/imports, CLI options and mode constraints, the
recorded regression upload's metadata, `models/best.pt`, the CV interpreter,
available disk space, an unused isolated output directory, and focused
automated tests. It only probes video metadata; it never processes video frames
or starts `main.py` on the long clip. Non-zero exit status means at least one
required check failed. A skipped full-video run is not evidence of real-video
accuracy or successful regression.

Defaults target the previously analyzed upload
`omniBoard/backend/local-data/uploads/737fda0c826f4a0bbbc6669830b7e6cd.mp4`,
the CV environment at `.venv-1`, the model at `models/best.pt`, and a new
timestamped directory under `output/pre_regression_runs/`. Override paths with
`--input-video`, `--model`, `--cv-python`, or `--output-dir` when appropriate.
The disk-space threshold is the larger of 1 GiB and three times the prior
tracked-video artifact size (241,247,583 bytes), providing room for temporary
segments and final output.

## Verified full-video command (run separately)

This PowerShell block uses the previously analyzed upload, the CV interpreter
and model above, online canonical IDs, and the two supported online-only
features. The timestamped output path must not already exist. No checkpoint
argument is supplied, so this run cannot resume from a stale checkpoint.
Confirm the hardware-selected `auto` backend in the run log before interpreting
performance.

```powershell
$runStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "C:\omniBoard-Sportz\output\pre_regression_runs\full_video_$runStamp"
if (Test-Path -LiteralPath $outputDir) {
    throw "Output directory already exists; choose a fresh path: $outputDir"
}
$env:IDENTITY_DIAGNOSTICS = "true"
& "C:\omniBoard-Sportz\.venv-1\Scripts\python.exe" `
  "C:\omniBoard-Sportz\main.py" `
  --video "C:\omniBoard-Sportz\omniBoard\backend\local-data\uploads\737fda0c826f4a0bbbc6669830b7e6cd.mp4" `
  --model "C:\omniBoard-Sportz\models\best.pt" `
  --output-dir $outputDir `
  --backend auto `
  --canonical-ids online `
  --ball-gate `
  --team-smoothing `
  --progress-every 100
if ($LASTEXITCODE -ne 0) {
    throw "CV regression run failed with exit code $LASTEXITCODE"
}
```

## Full-video evaluation record

Copy and complete this section after a separately approved full-video run.
Keep the input, model, output directory, mode flags, and environment together
so the run is reproducible. Do not treat identity or team metrics as accuracy
without labeled ground truth and human review.

| Field | Result |
|---|---|
| Run date / analysis ID | |
| Input video path / SHA-256 | |
| Source frames / dimensions / FPS / duration | |
| Python executable / version | |
| Model path / SHA-256 | |
| Inference backend / device / image size | |
| Exact command and mode flags | |
| Isolated output directory | |
| Final status | |
| Frames processed / output-video frames | |
| Model initialization / processing / total elapsed time | |
| Processing FPS | |
| Raw tracker IDs / canonical IDs / retained identities | |
| Association evidence rows / ambiguous links / proposed merges | |
| Ball observations: real / interpolated / missing | |
| Possession frames: Team 1 / Team 2 / Loose / Unknown | |
| Human-reviewed clips / sample method / reviewer | |
| Known failures, limitations, and follow-up | |

### Required artifact and integrity review

- [ ] `tracked.mp4` exists, is playable, and has the expected frame count.
- [ ] `observations.csv` exists, has a header, and covers every processed frame
      for ball observations.
- [ ] `pitch_rejected_detections.csv` exists; inspect its header and counts.
- [ ] In online mode, inspect `identity_association_evidence.csv` and
      `identity_transitions.jsonl`; report ambiguity rather than treating
      association scores as calibrated probabilities.
- [ ] In offline mode, inspect the canonical map, ambiguity/candidate artifacts,
      and reconciliation report; unresolved candidates remain unmerged unless
      independently verified.
- [ ] Confirm metadata/status and frame counts agree between the API/dashboard,
      worker result, and generated artifacts.
- [ ] Manually review representative identity switches, re-entries, occlusions,
      team assignments, ball gaps, and possession segments against the video.
- [ ] Record separately whether artifacts were produced and whether their
      contents passed human/labeled accuracy review.
