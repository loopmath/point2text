from point2text.diffusion.work import native_work_summary


def test_work_counts_actual_updates_including_early_stopping_not_requested_settings():
    summary = native_work_summary([
        {'request': {'steps': 32}, 'runtime': {'passes': 3},
         'guidance': {'config': {'inner_updates': 16}, 'steps': []}},
        {'request': {'steps': 32}, 'runtime': {'passes': 7},
         'guidance': {'config': {'inner_updates': 16},
                      'steps': [{'updates': [{}, {}]}, {'updates': [{}]}]}},
        {'runtime': {'passes': 4}, 'guidance': None},
        {'error': 'native operation failed; runtime receipt unavailable'},
    ])
    assert summary == {'trajectories': 4, 'native_passes': 14, 'guided_passes': 2,
                       'encoder_updates': 3, 'trajectories_without_runtime_receipt': 1}
