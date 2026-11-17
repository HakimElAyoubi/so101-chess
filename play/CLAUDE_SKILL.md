# Claude as the grasp and place skill (play/llm_skill.py)

The staged player (play/policy_player.py --variant skills, play/staged.py) runs a move as approach -> grasp -> carry ->
place -> return; the grasp and the place are "skills" with the interface of play/policy_io.Policy. `claude:<model>` as
`--grasp-repo` / `--model-repo` loads `ClaudeSkill`, which drives those two stages with Claude's vision in a
look-measure-move loop. Everything else (IK approach with sag removal, carry, the hold / miss / release rules, the
retries, logging, videos, trials.jsonl) is unchanged.

## Run
```bash
# the gate / a game (Terminal.app for the camera; the key from ANTHROPIC_API_KEY or ~/.anthropic_api_key)
bash play/restart_player.sh scripts/play_claude.sh
# dry run on the fake arm with the twin's truth instead of Claude (no API, no key): the whole loop except vision
python play/policy_player.py --variant skills --sync --fake --fake-render 0.004,-0.006,0.8 --port 8832 --model-repo claude:mock --checkpoint final --grasp-repo claude:mock --grasp-checkpoint final
# dry run with the real model on twin-rendered frames: the pieces and the printed squares stand on a board shifted by
# (dx m, dy m, yaw deg) from where the player believes it is, so the loop has to find and remove that bias
python play/policy_player.py --variant skills --sync --fake --fake-render 0.004,-0.006,0.8 --port 8832 --model-repo claude:claude-opus-5-5 --checkpoint final --grasp-repo claude:claude-opus-5-5 --grasp-checkpoint final
python play/claude_looks_montage.py play/results_dryrun/claude/<time>     # the looks of a trial on one sheet
```
Then in the page: "Measure board" is not needed on the fake arm; type a move ("custom: move the black pawn from e7 to e5")
and run it. Each look's annotated image, Claude's answer, the measurement and (twin runs) the truth land in
play/results*/claude/<time>/ (look_NN_from.png / look_NN_to.png, looks.jsonl).

## Division of labour: Claude points, the camera model measures
Claude is asked for ONE PIXEL per look and nothing metric: the centre of the target piece's top (grasp) or the centre
of the printed target square (place), plus flags (visible, clear, knight roll, drawn-vs-real fingertip offset). The
skill back-projects that pixel with the real fisheye model onto the plane where the pointed feature lives and compares
it with LAND, the point where the feature must be for a vertical descent to put the stem between the pads:
- LAND comes from the camera-to-jaw geometry only (the fingertips from the twin's FK at the measured joints, projected
  with the real intrinsics), so the twin's board / arm error (~8 mm at the hover) never enters the measurement.
- The pointed plane matters. From the wrist camera (about 29 deg off vertical at the hover) only the HEAD of a piece is
  visible; back-projecting the head onto the stem's height lands 4-5 mm beyond the stem (0.55 mm per mm of height).
  The first version asked Claude for "the stem" and for millimetres read off a grid: it pointed at the head, the grid
  reading drifted, and the pads closed 4 mm beside the pawn. Now the head centre is back-projected at the head's own
  height (TOP_Z_CM from the piece profiles in chess_sim/assets/pieces/manifest.json: pawn 2.15, rook 2.7, bishop 3.3,
  queen 3.7, king 4.5 cm; the knight's top is off its axis, so its round base on the board is used instead).
- A reported fingertip offset (real minus drawn, up to 25 px, confidence >= 0.5) shifts LAND by the same amount on the
  tips' plane: a running hand-eye check on the real arm.

## The loop
- Settle before every look (joints within 0.25 deg of the target for 0.35 s): the P=16 servo lags 100-240 ms. The
  command is desired + an integral bias (the player's own sag removal), so the measured pose is what Claude saw.
- Hover: look; move the fingertips by the measured offset (numerical Jacobian on the measured joints, tilt held,
  min-jerk 0.7 s, at most 15 mm per look) while it exceeds 2 mm, at most 3 moves; then descend to the humans' close
  height for the piece (drill medians: pawn 1.35, rook 2.03, knight 1.71, bishop 2.27, queen 2.81, king 2.68 cm above
  the board) at 25 mm/s with the jaws at 12 %.
- Bottom: look again, a last correction when the offset exceeds 2 mm (at most 2), then the 0.8 % close; the player's
  stall rule confirms the hold or retries.
- Give-up (the piece is not in view, the way down is blocked, or still off after 3 moves): the player retries at once
  (play/staged.py reads the skill's `gave_up` reason) instead of waiting for the 90 s timeout. 5 looks per attempt.
- Place: the held piece's base against the printed square's centre (the ring is only the belief; if Claude cannot find
  the square the piece goes down at the ring, which is within ~10 mm), lower to 3 mm, release at 14 %.
- Knights: Claude gives roll_deg to close across the neck.

## Model and cost
claude-opus-5-5 with adaptive thinking (it cannot be switched off on this model) at effort low (`CLAUDE_EFFORT` in the
launcher's environment: low|medium|high; a shell that carries another value changes every look, which is how the first
dry runs ended up thinking 3000 tokens per look and timing out). The system prompt is cached (about 1.3k tokens), the
answer is a 60-token JSON; a look costs about 1.1k input + 0.3-0.6k output tokens, roughly $0.02. The 3000-token first
attempt is retried once with 8000 tokens at low if the answer is cut. `CLAUDE_MODEL=claude-sonnet-5-5` switches the model.
Mock dry run on the shifted twin board (2026-11-10): 3 looks per stage, one correction each, placed in 22 s; the
measured offsets equal the truth to 0.1 mm (the back-projection is the inverse of the projection by construction).
Real model on twin frames: see the measured section below.

## Real arm, 2026-11-10 (first trials, all missed at the close; fixes above came from them)
- Heights: the humans' close heights and the player's close_h are heights of the planner's grasp site (Planner.fk),
  6 mm above the fingertip sites; the descent now puts the site there. The point Claude points at has a height too
  (TOP_Z_CM): every cm it is wrong moves the landing point ~5 mm along the camera's view axis, which the monocular
  bottom look cannot see (the piece still appears between the pads); measure the real pieces and write
  data/piece_tops_cm.json.
- At the player's hover the sponge pads hide the top of a tall piece: the skill looks from 3.5 cm above the top first.
- No goal marker in the image: with the LAND circle drawn, Claude reported the target at the circle (bottom looks of
  the queen: crown 4-6 mm away). The outline of the target square is drawn twice for a grasp (printed square, dashed;
  lifted to the tops' height, solid, with an arrow), because Claude read the lifted outline against the printed letters.
- Claude's `clear` flag fires on the pads themselves: ignored. Jumps of 8+ mm after a converged look are refused.
- Retries probe the hand-eye along the jaw axis (+3, -3 mm); a hold on a probe gives HAND_EYE_MM.
- The trial videos (play/results/videos/<time>_<ckpt>_<n>.mp4 + _joints.npy at 30 rows/s: time, 6 measured, 6
  commanded) show the close itself: frames at the close command are the ground truth the looks cannot give.
- Looks take 5-10 s on the real frames at effort low; a stage attempt fits 4-6 looks in its 90 s.

## Tuning
The prompt is SYSTEM_PROMPT in play/llm_skill.py; the constants at the top (openings, heights, tolerances, moves per
phase). looks.jsonl has, per look, Claude's pixel, the measured offset (d_mm), the truth (truth_mm, twin runs), the
action and the correction applied; play/claude_looks_montage.py draws them on the images.
