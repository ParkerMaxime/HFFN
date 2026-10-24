# Run stopped by the agent after the sweep stage

The sweep stage completed (12 jobs, all exit 0; `sweep/selection.json`). The best rate of both arms (`E0` and
`E3b`) is the **last point of the grid (0.016)**, so by the rule of `README.md` ("if the best rate of an arm is at an
edge of the grid, the sweep is declared inconclusive and repeated on a grid extended by two points") the sweep is
inconclusive and the final stage at that rate must not stand. The launcher (`lib.campaign`) was killed by the agent at
18:28 UTC right after it started the final stage, and the four `final/E0_s*` jobs it had just started were killed
(SIGTERM) a few seconds later: they have no `summary.json` and no figure. A new run of the step follows with the grid
extended to {0.032, 0.064}. Not a candidate failure; no verdict is drawn from this folder except the inconclusive
sweep.
