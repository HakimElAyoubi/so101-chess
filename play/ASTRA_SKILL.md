# Astra grasp and place

Run `scripts/play_astra.sh` from the repository root.
It runs **GPT-6.1 Terra, low effort, standard pricing for grasping only** (switched from Sol on 10 November 2026: $1/$5 per million tokens instead of $2/$10, Roboflow Vision Evals 85.8 % vs 81.7 %, faster answers; `ASTRA_MODEL=gpt-6.1-sol` restores Sol).
Approach, carry to the target square, lowering, release and return use calibrated
scripts. Placement makes **zero vision/API calls**. The command retains its old
Astra name for compatibility.
Stockfish chooses moves. The existing staged player performs approach, carry,
return, sag compensation, motion limits, hold detection and retries.

Supply the API key through `OPENAI_API_KEY` or `~/.openai_api_key` (one line,
permissions 600). No OpenAI Python package is required. API requests send the
wrist crop to OpenAI with `store: false`.

If replacing an existing player, run from Terminal:

```bash
cd "$HOME/Desktop/SO 101"
bash play/restart_player.sh scripts/play_astra.sh
```

In the player: ready pose (H), **Measure board (automatic)**, check the grid (G),
then choose a trial. Do not issue calibration/nudge commands during a trial.

## How the revised skill works

1. At the settled hover, Astra identifies the named piece's top centre (the
   knight's base) and the real pad gap when both contact tips are visible.
   The crop has one thin square boundary, with no target dot, dense grid,
   arrow, predicted pad marker or text across the piece.
2. The camera model converts pixels to millimetres at the feature's own height.
   A bounded pad correction (at most 6 mm) accounts for the real gap differing
   from the twin. It can be applied at the hover and bottom; no fixed board
   translation or speculative gripper constant is introduced.
   A fresh Astra `between_pads=true` response can close immediately when the
   intended body is at the pads' contact height, even if the top pixel is hidden.
   This contact judgment takes precedence over further pixel centering/descent.
3. A local tracker can verify up to two corrections without another API call.
   Physical black queen hover corrections reacquire Astra to remeasure the pad gap
   after pose/height changes.
   Its original 33-pixel patch stays fixed to avoid cumulative drift. It requires
   a correlation of at least 0.80 and a unique peak separated by at least 0.08;
   weak texture, excessive movement, frame edges, ambiguity, changed appearance,
   or age over 10 seconds force an Astra reacquisition.
4. Body grasp heights use the human demonstration medians in `llm_skill.py`,
   except the black queen, lowered 11 mm following physical user feedback. Pointing at the head identifies its axis; the jaws descend
   to the body instead of clamping the crown. Normally a fresh Astra look is required
   after the descent/opening change; tracking never bridges that transition.
   The black queen and bishops instead use the guarded descent-and-close procedure below
   when the last hover was clearly verified, because the pads can hide its crown.
5. The existing OpenCV X-corner/PnP detector handles printed board corners when
   its fit is well supported. Astra no longer enumerates corners, computes a
   pad displacement, or writes commentary on every look. Place centres are
   reacquired with Astra rather than tracking the carried piece.
6. Uncertain or blocked measurements cannot authorize a descent, close or release.
   The hover still aims within 2 mm. At body height, a clear target with confidence
   at least 0.85 can authorize a close inside a bounded capture window: 4 mm along
   the closing axis, 2 mm across it, 4 mm overall, and measured site height within
   3 mm of the demonstrated body height. This allows the closing stroke to make
   contact instead of chasing camera errors below the calibration uncertainty. Exhausting the look budget cannot close the jaws blindly.
   A response from an expired attempt cannot replace a new attempt's answer.

The stage timeout is 180 seconds, including API time; the player's existing
300-second whole-move cap remains. There are at most eight measurements per
attempt, including local measurements. Motion speeds and joint limits are
unchanged. The higher stage guard prevents a slow but useful look expiring
immediately before closing. Premium Fast processing is disabled by default to reduce cost.

## Settings, costs and logs

- `ASTRA_EFFORT`: `low` (default), `medium`, `high`, `xhigh`, `max`.
- `ASTRA_SERVICE_TIER`: `default` (default), `fast`, `priority`, `ultrafast`.
  Fast processing costs **2x standard API rates per call**. Tracking reduces
  repeated calls, but the net cost per successful move is not yet measured.
  Use `ASTRA_SERVICE_TIER=default` to retain standard processing.
- `ASTRA_MODEL`: default `gpt-6.1-terra` (`gpt-6.1-sol`, `gpt-6-astra` also priced in the ledger).
- Local estimated limits: $0.10 per move, $1 per launcher process, 12 requests
  per move including retries. Shared ledger: `data/logs/astra_api_usage.jsonl`.
  Input reservation is conservative; these are local estimates, not account billing limits.
- Responses are limited to 1024 output/reasoning tokens. Truncation is recorded
  and not automatically resubmitted at a larger token budget.
- Log: `data/logs/astra_player.log`; progress: `data/progress_astra.txt`.
- Looks: `play/results/astra/<timestamp>/`; fake runs: `play/results_dryrun/astra/`.
  The answer's `_source` distinguishes `astra` from `local_tracker`; `_service_tier`
  records the tier returned by the API. Local updates have zero API token usage.

## Checks and evidence (10 November 2026)

The failed queen trial `20261110_153210` spent 29–33 seconds per look; the
120-second stage guard expired just after three hover looks and descent.
The recorded inputs were replayed without opening the robot serial port.
On three identical reconstructed video crops, the revised prompt at medium
measured 22.34/33.74/32.38 seconds with standard processing and
13.80/14.78/21.65 seconds with confirmed Fast processing. This is a small replay
benchmark, not a hardware success-rate measurement. The archived video is
640x360 and was restored to the camera coordinate size before cropping.

The anchored tracker followed 523 sampled frames of the recorded queen clip
in an offline replay (about 0.15 ms per tracking update). Age checks use live
wall time; the offline replay processes frames faster than their recorded rate.
The board detector declined that occluded view instead of inventing corners.

Run the offline regression suite (no hardware and no paid API calls):

```bash
python play/check_astra_skill.py
```

It covers API contracts, invalid measurements, API failures, truncated answers,
old-worker rejection, tracking/reacquisition, look-budget exhaustion, an anchor
without a grid residual, and complete simulated pawn/knight/queen moves.
Simulated success does not establish real-arm grasp accuracy; the next real
trial must verify the changed pad correction and target localization.

For the interactive twin without an API key:

```bash
python play/policy_player.py --variant skills --sync \
  --fake --fake-render 0.004,-0.006,0.8 --port 8832 \
  --model-repo astra:mock --grasp-repo astra:mock
```

Journal basis: 27–28 October (gravity sag, path clock, demonstration heights,
no double application of height maps), 30 October (clamp the body; real gap is
not the twin gap; shadows can fake corners), 1–2 November (read square identity
from the board, not overlays; pose-dependent gripper error is not a global board
translation). These lessons informed the changes; calibration files and gains
were not rewritten.

Sources: [Astra model and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Fast mode](https://developers.openai.com/api/docs/guides/fast-mode),
[latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization).

### Follow-up: pawn aligned but no close (10 November, 16:00)

The e7-e5 trial `20261110_155904` never sent a close command. The first body-height
look was clear, confidence 0.93, at 3.00 mm along the jaw axis and 0.12 mm across,
with site height only 0.10 mm from the body target. Two subsequent nudges left a
3.06 mm radial estimate, and the old strict 2 mm close rule gave up.
The capture window above now sends the close at that first verified bottom look.
All 17 offline checks pass, including replaying all three recorded bottom
measurements and testing rejection outside the capture window. This demonstrates
that the command is sent; a successful physical pickup still needs a new trial.

### Knight quarter-turn (10 November, 16:10)

The user confirmed the pawn pickup succeeded, but the knight needs the grasper
turned 90 degrees before gripping its narrow neck. Both the system prompt and
per-look knight text now state that requirement explicitly. The controller
performs a single equivalent +90/-90-degree wrist turn above the piece, rather
than accepting changing incremental roll guesses during alignment. Afterward,
Astra is told the turn is complete and `roll_deg` is held at zero so another
look cannot undo it.

The turn is independent of lateral alignment: it happens even when the knight
is already centred. Before turning, the fingertips are raised to 20 mm above the
knight's physical top (not its pointed base). Planning preserves fingertip
position, uses the existing joint-limit/tilt fallbacks and 30 deg/s peak motion,
and gives up if no reachable quarter-turn exists. It runs once per grasp
attempt; pawn and other piece behavior is unchanged. The session log prints
`knight quarter-turn: -90 degrees before visual alignment` (or +90).

Nineteen offline checks pass, including both b8/g8 quarter-turns, joint bounds,
under-2-mm fingertip displacement, no repeated/undone turn, and a full simulated
knight grasp/carry/place. The new real-arm knight pickup is not yet verified.
Restart the player to load this change.

### Black queen: lower body height and close after verified descent (10 November, 16:30)

The user reports that the black queen is correctly aligned but slightly too
high and never closes. Its Astra site target is now **2.51 cm above the board**,
down from 2.81 cm. Other pieces and the white queen keep their existing heights.
The prompt explicitly asks for the black queen's body below the crown.

The recorded queen trial `20261110_162334` also exposed a false relook loop:
an old grasp anchor was treated as enough evidence to reject a confidently
identified queen as 15-17 mm from its square. Astra now requires a current,
accepted grid fit before that identity veto can apply. Fewer than three
non-collinear matched corners cannot update the registration; an old anchor
alone cannot veto the next piece.

For a clear, visible queen with hover confidence >=0.90 and residual <=2 mm,
the controller records the aligned axis and descends straight to the lower
body target. It solves using the commanded 15% opening, rather than the 10%
hover opening, and refines the IK. It then closes when the joints have settled,
site height is within 1.5 mm of target, and fingertip XY drift is <=2 mm.
It does not request another crown pixel after the pads hide it or make new
lateral guesses at the bottom. Failure to settle within five seconds of the
planned descent ends the attempt. Hold/miss detection and retries still follow
the close. Without the confident hover evidence, this automatic close is not
authorized and the normal visual path remains.

All 24 offline checks pass, including the saved aligned queen frame, lowered
height, no extra API look before closing, wrong-height rejection, and stale/
sparse grid rejection. The simulated queen move completes without retries at
about 2.50 cm measured site height. This new physical queen pickup is unverified;
restart the player before trying it.

### Black queen: another 2 mm lower (10 November)

The user reports the previous change works much better and requests a further
reduction. The current Astra black queen grasp target is **2.31 cm above the
board**, another 2 mm lower than 2.51 cm (5 mm below the original 2.81 cm).
Pawn, knight, and white queen heights remain unchanged. The verified descent
and close checks use this updated height.

Validation: all 24 existing offline checks passed at this height. The new height
has not yet been tested on the physical robot. Restart the Astra command to load it.

### Black queen: another 6 mm lower (10 November)

At the user’s request, the current Astra black queen grasp target is now
**1.71 cm above the board**, another 6 mm below 2.31 cm (11 mm below the
original 2.81 cm). Other piece heights and the descent/close checks are unchanged.

Validation: all 24 offline checks passed. This height awaits a physical trial;
restart the Astra command to load the updated policy.

### 10 November — queen close path bypassed at 88% confidence

The user reports correct alignment, no grasp, then forward movement tilting the
piece. Trial `20261110_164404` converged at hover to (-0.59,+1.41) mm with
confidence 0.88 (local tracker score 0.935). The direct descent/close path
required 0.90, so it silently fell back to bottom camera checks. Those commanded
(+3.4,-6.5) mm, then (+2.1,-2.9) mm before losing the queen; no close was sent.

The queen path now accepts a clear visible hover at confidence >=0.85 and <=2 mm
residual, consistent with the existing body capture confidence floor. Unverified
queen hovers stop before descent instead of entering the bottom correction loop.
Refined descent explicitly holds the certified hover XY; a defensive guard
prevents any lateral black queen correction at body height. The close still
requires settled measured pose, site height within 1.5 mm, and XY drift <=2 mm.
Grasp height stays at the requested 1.71 cm above board. Prompt updated.

All 27 offline checks passed, including replay of the actual 0.88-confidence
hover through descent and close without a bottom look, rejection of uncertain/
blocked hovers, and rejection of lateral corrections at body height. Simulated
queen placed with no retries, 3.62 mm placement error; pawn/knight regressions
passed. Physical behavior awaits restart and a new trial.

### 10 November — black queen centering: 5 mm toward the H-file

The user reports a grasp left of centre and explicitly requests 5 mm toward the
H-file, describing this as left from their physical viewpoint. This supersedes
the proposed 2 mm correction. Astra now adds a 5 mm A-to-H board-file-axis trim
to the physical black queen hover target. It is an absolute target offset, not
an accumulating movement. The existing hover correction positions the arm before
descent; no lateral correction is introduced at body height. The requested
1.71 cm grasp height and verified descent/close checks remain unchanged.

The trim is physical calibration only: other pieces, white queen, placement,
Claude and idealized twin runs do not receive it. Each measurement logs both the
file-axis trim and its world-coordinate vector. The software applies it directly;
the model is not asked to estimate millimetres or reinterpret image left/right.

Validation: all 28 offline checks passed, including the 5 mm H-file direction,
non-accumulation, and scope exclusions. A physical trial of this trim is pending;
restart the Astra command to load it.

### 10 November — H-file trim must finish below the high camera pose, above the crown

The user reports another failed grasp. Trial `20261110_165745` sent no close:
the first attempt exhausted four hover corrections, ending at a 2.49 mm residual;
the second encountered an unreachable lateral correction. The high camera pose
held wrist tilt near its limit while chasing the requested 5 mm H-file offset.
The measured pad offset also crossed its 6 mm application threshold in the last
look of the first attempt. No evidence here indicates a failed close motor command.

The 5 mm H-file trim is retained, but removed from high-hover visual residuals.
The recorded fourth look now measures within 2 mm without that trim. Once an
unshifted hover is verified, the controller plans both waypoints before moving:
finish the 5 mm trim with tips 10 mm above the taller of the queen mesh/crown
height, settle and verify within 2 mm, then descend vertically on that shifted
axis to the unchanged 1.71 cm site height. Geometry uses the actual open grip.
Both endpoints must be reachable and within joint limits. An unsettled clearance
waypoint times out without descent or close; body-height lateral corrections
remain prohibited. Final close still checks settled height and XY drift.

All 31 offline checks passed: recorded failure measurement replay, physical
5 mm trim planning from two recorded hover poses, sampled joint-path crown
clearance, no premature close at clearance, final close, clearance failure,
and pawn/knight/Claude regressions. Idealized twin runs do not apply the physical
trim; the trim-specific tests explicitly enable it. These checks do not establish
physical grasp success. Restart and a new hardware trial are still needed.

### 10 November — alignment changes reverted at user request

User: “forget the alignment, go as it was before.” Removed the 5 mm H-file
trim and the extra above-crown waypoint. Restored the earlier direct descent
on the verified hover axis and settled close at 1.71 cm above the board.
The confidence 0.85 / residual 2 mm hover checks, refined open-grip geometry,
and prohibition on lateral body-height corrections remain as before the offset.
Previous trim entries above are historical and no longer describe active behavior.

All 30 offline checks passed, including physical-mode replay verifying the
unshifted hover axis, no extra approach step, and close after direct descent.
Restart the Astra command to load this restored policy.

### 10 November — descent planned from unachieved wrist command; use measured pose

After reverting alignment changes, the queen still failed. In trial
`20261110_170850_final_12`, the hover converged to 0.69 mm (confidence 0.89,
look 2 in `astra/20261110_170906`), but no body close was logged. Joint telemetry
shows actual wrist flex 101.978 deg while the command was 105.978 deg. The arm
was not achieving the commanded reference. Replaying descent from that command
reproduces `cannot preserve queen alignment at lower body height`; replaying
from the measured encoder pose reaches the requested body height within joint
limits. This explains an immediate retry after a logged descend decision.

Astra now records the current measured joint pose on each tick. At the verified
black queen hover-to-descent transition, it rebases the planned descent on that
physical pose and clears the previous reference's accumulated sag integral.
The certified XY also comes from that measured pose. This prevents an unreachable
wrist reference and its accumulated error from being reused for the next path.
The 1.71 cm height, zero file offset, direct descent, and settled-close checks
remain unchanged. A rejected descent now also prints its reason in the log.

All 31 offline checks passed. A new regression retains the failed actual/command
pair, reproduces the old rejection, then verifies the measured-pose descent,
reachable endpoint, reset integral, unchanged measured XY, and final close.
This is evidence for the control fix, not physical grasp success. Restart the
Astra command before a new physical trial.

### 10 November — user-reported success after roughly ten retries; reliability unresolved

The user reports the queen finally grasped after “like 10 retries.” This is a
physical success report, not evidence of reliable first-attempt performance.
The latest process session (`astra/20261110_171630`, launcher 17:15:47) records
three attempts: first hover converged to 1.29 mm but confidence 0.82 failed the
0.85 queen descent gate; second exhausted hover corrections at 4.28 mm residual;
third converged to 1.38 mm with confidence 0.90 and commanded close. Measured
close site was 1.72 cm above board, with +0.1 mm height error and 0.23 mm XY drift.
Earlier retries span previous sessions, so the user's approximate count is not
an exact per-policy success-rate denominator. No completed move/carry success
record is available for this latest session.

This supports the measured-pose descent fix reaching a real close, while hover
convergence and confidence rejection still cause restarts. Kept the current
1.71 cm target and zero alignment offset as the physical-success baseline.
No further controller tuning or hardware operation was performed in response
to this outcome report. Future improvements must distinguish confidence rechecks,
hover convergence failures, empty closes, and losses during carry.

### 10 November — bounded hover recovery for repeated queen retries

The user requested a reliability fix after reporting success only after roughly
ten retries. Changes target the recorded confidence and hover-reach failures,
without adding an alignment offset or changing the 1.71 cm body grasp height.

A converged queen hover below 0.85 confidence now requests a fresh model look
in place, with at most two rechecks and the existing eight-look / 180-second
limits. It cannot descend on the low-confidence observation. A clear confident
hover still 2–6 mm away after four corrections may make at most two additional
corrections within the same attempt; the 2 mm descent tolerance is unchanged.
Blocked, missing and unconfirmed measurements still cannot authorize descent.

Physical queen hover correction IK now starts from the measured pose, checks
joint limits, and resets the obsolete pose-error integral. If the same-height
correction is unreachable, it tries 5/10/15 mm lower hover targets, retaining at
least 15 mm above the taller of the queen mesh/crown height. It preserves tilt
and roll, checks sampled joint-path clearance, limits motion speed, and verifies
both the next hover and subsequent body descent are reachable in the regression.
Only if necessary does it shorten the lateral correction. It reacquires the pad
gap after corrections instead of carrying a pad-offset estimate across changed
pose/height. No lateral body-height corrections or fixed file offsets are added.
Physical queen commands are bounded to calibrated joint limits; anti-windup
removes the rejected part from the integral so it cannot build up at the wrist
stop again. Claude and other piece controllers retain their prior behavior.

All 37 offline checks passed. Regressions cover the real 0.82-confidence hover
(recheck then verified descent), bounded rechecks, blocked/exhausted-look
refusals, the recorded 4.28 mm hover-budget failure, and full corrections from
six recorded poses. Each of those six reaches its requested XY within 0.6 mm,
keeps the sampled hover path above the crown, stays within endpoint joint limits,
and admits a subsequent body descent. A repeated-command check verifies no
wrist-limit integral accumulation. Existing pawn/knight/queen and Claude twin
checks passed unchanged. These are geometry/control checks, not a measured
first-attempt success rate. This version requires restart and a physical trial.

### 10 November — bishop top occlusion prevented any close command

The user reports the bishop arrives correctly but never grasps. In session
`astra/20261110_180458`, black bishop c8 hovers converged to 0.62 mm at confidence
0.87 and 1.26 mm at confidence 0.92. Both descended, then received invisible /
null target / confidence 0.35 observations at body height and immediately gave
up. Neither attempt sent close. This is the same top-occlusion failure previously
handled only for the black queen, rather than evidence of an empty closed grip.

Generalized the verified descent certificate from black queen to bishops of
both colours. Bishops keep their demonstrated 2.27 cm site target. A visible,
clear hover with confidence >=0.85 and residual <=2 mm is retained; the descent
is solved from measured joints at the commanded open grip, with XY held and
joint limits checked. At the bottom, close is authorized by settled measured
site height within 1.5 mm and tip-axis drift <=2 mm, without requesting another
view of the now-occluded top. The five-second settling deadline and normal
hold/empty-grip detection remain. No body-height lateral corrections are allowed.
Bishops also receive the bounded confidence recheck and command-limit anti-windup
logic; queen-specific extra hover corrections and lowering remain queen-only.
No bishop height change or fixed alignment offset was introduced.

All 41 offline checks passed. Both recorded bishop hover poses were replayed
in physical mode for black and white eligibility, reaching a bounded refined
descent and close without a bottom model look. Tests reject unverified hovers,
wrong-height poses, and body-height lateral corrections. Added bishop c8-h3 to
the full simulated-move checks: Astra placed with no retries, 3.89 mm placement
error, 31.0 simulated seconds; Claude bishop also passed. Existing queen, pawn,
knight and API checks passed. The new bishop behavior awaits restart and a real
trial; offline replay does not establish physical grasp success.

### 10 November — direct visual body-between-pads close; API outage handling

User requested: grasp when the piece is between the clamps. Latest session
`astra/20261110_181215` did not reach the bishop: after two hover measurements,
repeated OpenAI HTTP 429 errors exhausted the visual budget across retries.
Those were transport failures, not evidence that the new bishop descent missed.

Added strict `between_pads` to Astra's JSON schema and prompt. It means the
intended piece's graspable BODY is physically between both facing contact pads
at their current height, so closing now would grip it. It is false for a piece
merely projected into the image gap below an elevated gripper. Confidence rates
this contact/identity judgment when true; a hidden top can have visible=false
and target_px=null while the body judgment remains true.

A fresh Astra response with between_pads=true, confidence >=0.85, clear=true
and both-pad midpoint available now commands CLOSE immediately at the current
pose. It bypasses top-pixel measurement, further alignment, demonstrated-height
vetoes and further descent. It applies to all piece types during grasp only.
A local tracker may never reuse a positive contact judgment; current joints
must remain within 1.5 degrees of the pose in the fresh image. Existing
hold/empty-close checks still verify whether the grasp actually caught anything.
The earlier verified descent remains available for approach when no contact
judgment has yet been obtained. Existing heights and zero file offset remain.

HTTP 429 now produces a nonblocking hold with bounded exponential backoff and
jitter, honoring valid Retry-After delays. At most three retries and sixty
seconds of recovery are allowed within the stage deadline. Each retry captures
a fresh image rather than reusing a delayed contact observation. Transport
errors do not consume visual look slots. Quota/billing/access failures, long
server retry delays, and persistent errors stop the trial holding position,
without triggering arm retreat/retry cycles. Only sanitized status/flags are
logged; raw server messages and authorization headers are not exposed.
Implemented using official OpenAI guidance:
https://developers.openai.com/api/docs/guides/rate-limits

All 47 offline checks passed. New checks cover immediate close for all six
piece types with hidden tops, no extra centering/descent, rejection of cached/
uncertain/moved-pose/wrong-stage contact signals, body-contact schema validation,
Retry-After handling, fresh-frame reacquisition, bounded persistent-429 stopping,
quota classification, and no automatic arm retry on fatal API errors. Prior
recorded bishop/queen regressions and full simulated moves also passed.
No live API request or hardware move was made for these tests; this direct
visual trigger still needs a physical trial after restarting the command.

### 10 November — cheaper grasp vision; fully scripted target motion and placement

User reported roughly $10 spent and requested a lower cost with similar
performance, followed by explicit instructions that only grasping use vision.
The cost audit found 95 recorded paid Astra Fast readings with 140,880 input
and 52,997 output tokens (no cached input), approximately $8.1173 at the current
$20/$100 per-million Fast rates. Three additional paid readings have unknown
tiers; development replays and other account usage are outside that subtotal.
The robot was not sending 30 camera frames/second to the API: it sent settled
look requests, but premium pricing, reasoning output and repeated attempts added up.

The existing command now defaults to GPT-6.1 Sol, low effort, standard service
for grasping, with full-resolution crops and all geometric/contact checks kept.
Standard Sol input/output rates are $2/$10 per million: 90% below Astra Fast
per token. Local tracking remains free. The response cap is 1024 tokens with
no automatic expensive truncation resubmission. A shared process ledger charges
all responses, including incomplete ones, and reserves before issuing a request.
Local estimated caps are $0.10/move and $1/launch, plus 12 requests per move;
retries do not reset a move's allowance. Failures/unknown billing reserve
conservatively where needed. Budget exhaustion stops holding instead of cycling
the arm. Each request's estimated price and move/process totals are printed
and saved to `data/logs/astra_api_usage.jsonl`. These estimates are not an
account-level billing limit and a new process resets the launch allowance.

Only the grasp stage invokes vision. The player now loads `scripted:place` for
placement and `astra:gpt-6.1-sol` for grasp. Approach, carry above the target,
lowering, release, lift and return are scripted. The new place-only policy uses
the calibrated board target and measured holding height, plans the complete
Cartesian descent before moving, and opens only once the measured pose has
settled near the target. Unreachable/missing-height plans keep the grip closed.
It does not reuse the dry-run stand-in that would silently skip failed IK samples.

A bounded paid replay completed three Sol-low recorded queen/bishop images for
about $0.020602, versus $0.32488 estimated for their earlier Astra Fast responses
(~93.7% less on this small sample). Target differences were ~13.9 px for the queen
and 2 px for the bishop; the occluded bishop returned a body-between-pads signal.
The fourth request returned 429 and the comparison stopped without further calls.
This is a small static comparison, not proof of equal physical performance.
Raw comparison results are saved in the task's `outputs/Cheaper-vision-comparison.json`.

All 52 offline checks passed. New tests verify budgets across retries/new moves,
pre-request refusal, charging truncation without resubmission, cheaper defaults,
scripted placement failure holding closed, and complete pawn/knight/queen/bishop
moves with zero placement API calls. Those four scripted-place twin moves all
placed with no retries; idealized position errors were below 0.05 mm and do not
predict physical placement accuracy. Existing vision/control regressions passed.
Launcher shell syntax checked. New settings require restarting the same command;
physical grasp/placement performance still needs a trial.

Official pricing: https://developers.openai.com/api/docs/pricing
Model documentation: https://developers.openai.com/api/docs/models/gpt-6.1-sol

### 10 November — release when settled inside the drop zone

Screenshot/trial `20261110_183735_final_20` showed a held white bishop already
at the drop zone, but scripted placement failed its precise height condition.
Recorded final XY was 1.4 mm from the selected drop center. Reconstructing the
holding height from telemetry gives a release-height residual greater than the
2 mm square-placement tolerance, despite the piece being near the surface.

Drop-zone placement (identified by StagedMove.ring) now uses an area envelope:
15 mm radius around the selected drop spot and inferred piece-base height
-5 to +15 mm relative to the configured surface. It holds the reached measured
pose immediately and clears the old sag integral, rather than pushing toward
an exact height. After 0.3 seconds of physical stability within 0.3 degrees,
it opens and the existing release-confirmation/lift/return sequence completes.
Outside the area, too high/low, or still moving, it keeps holding. Normal square
placement retains its original 3 mm XY / 2 mm height checks. No API calls added.

All 56 offline checks passed. Tests replay the recorded failure pose and verify
opening, holding position rather than pushing, unchanged square precision,
refusal outside the drop envelope, and waiting for motion to stop. A complete
simulated bishop f1-to-drop-zone move placed with no retries and zero placement
API calls. The reconstruction fixture labels its holding-height estimate; it
is not a new physical calibration. Restart is needed for a physical trial.
