# Restart validation

The restart implementation and subsequent cache fixes passed **353 tests**, with
one optional real-encoder test skipped locally. GitHub's Linux/macOS package
checks and separate pinned-encoder parity job passed. Tests cover independent
basin progress, a monotone global winner, duplicate reactivation with current
quality approval, best-candidate ordering inside a cached cohort, raw-text
quality rejection, shared budgets, cancellation and cooperative deadline cleanup.

A browser fixture checked the requested 6/16/0.01/120 defaults, restart controls,
dependent-control disabling, live basin labels and completed work counts. Two
successive searches recorded different seeds. This validates the interface;
the fixture supplied corpus text and did not run a generator.

Two actual HTTP searches then ran on the installed wheel, at the original frozen
map's previously fixed off-corpus targets 01 and 04. Each requested one initial
candidate, one extra restart, one encoder update per guided pass, tolerance
`1e-4` and a shared 30-second deadline. Both entered a fresh restart, preserved
strictly improving global histories and completed without runtime errors.
They streamed 23 and 38 draft updates. Observed phases included coarse
generation, native repair and fresh restart generation. Actual elapsed times
were 30.36 and 30.16 seconds; running forwards can finish just after the deadline.

The [summary](experiments/hosted-restart-smoke.json) records requests, actual
seeds, basin transitions, executed work and remaining error. The
[complete run records](experiments/hosted-restart-smoke-runs.json.gz) retain
text and native traces. These are lifecycle checks with mechanical text
validation, not quality approval or epsilon successes. The separate
[six-target study](optimization-progress.md) applies independent blinded review.

The demo uses a regular installed wheel and preserves its map, run cache and
paragraph library across service updates. All three existing demo endpoints
returned ready after deployment. No Tailscale routing changes were required.
