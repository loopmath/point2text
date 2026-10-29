"""Summarize executed work from native receipts, never requested settings."""


def native_work_summary(trajectories):
    """Count observed denoising passes and completed encoder update records.

    Proposal vocabulary probes, quality models and model loading are separate
    costs. An errored operation without a runtime receipt has unknown work;
    it is counted explicitly rather than interpreted as zero work.
    """
    summary = dict(trajectories=0, native_passes=0, guided_passes=0,
                   encoder_updates=0, trajectories_without_runtime_receipt=0)
    for item in trajectories:
        summary['trajectories'] += 1
        runtime = item.get('runtime') or {}
        passes = runtime.get('passes')
        if type(passes) is int and passes >= 0:
            summary['native_passes'] += passes
        else:
            summary['trajectories_without_runtime_receipt'] += 1
        guidance = item.get('guidance') or {}
        steps = guidance.get('steps') or []
        summary['guided_passes'] += len(steps)
        summary['encoder_updates'] += sum(len(step.get('updates') or []) for step in steps)
    return summary
