# Run interrupted by a restart of the container

The container of the cloud session was restarted while this run was timing the cell b2 (cells a1-a4 and b1 had finished their five processes, b2 had not started
a process; c1-c2 not started). The launcher was killed with it; no `EXIT` was written by it (the file `EXIT` here, 137, was added by the agent afterwards).
Not a candidate failure and not a result: **no figure is drawn from this folder** (it is a partial run, and `delta` is defined over the cell-processes of one complete
run). The step is relaunched in full as a new run; the README and its criteria are unchanged.
