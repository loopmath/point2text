import json

import pytest

from point2text.diffusion.sentence_edits import sentence_spans, protected_repair_prompt
from point2text.profile import PromptPolicy


def test_scientific_boundaries_preserve_abbreviations_decimals_and_exact_offsets():
    text = ('  Fig. 2 shows a value of 1.25 for the first group. '
            'For example, e.g. a second measurement, uncertainty remains. '
            'Dr. A. Smith uses U.S. data. Unfinished fragment')
    spans = sentence_spans(text)
    assert [text[start:end] for start, end in spans] == [
        'Fig. 2 shows a value of 1.25 for the first group.',
        'For example, e.g. a second measurement, uncertainty remains.',
        'Dr. A. Smith uses U.S. data.',
    ]
    assert all(0 <= start < end <= len(text) for start, end in spans)


def test_quoted_sentences_and_repeated_sentences_have_distinct_source_spans():
    text = 'The authors call this "a conditional result." The result holds. The result holds.'
    spans = sentence_spans(text)
    assert [text[start:end] for start, end in spans] == [
        'The authors call this "a conditional result."', 'The result holds.', 'The result holds.']
    assert spans[1] != spans[2]


def test_empty_fragments_and_numeric_punctuation_are_not_sentence_proposals():
    assert sentence_spans('  ') == []
    assert sentence_spans('A fragment without final punctuation') == []
    assert sentence_spans('1.25') == []


def test_repair_source_is_quoted_data_and_keeps_the_use_case_and_qualifications():
    text = 'The hero said "it may rain." The journey continues.'
    span = sentence_spans(text)[0]
    policy = PromptPolicy(domain='fiction', content_rules='Keep the fictional world consistent.')
    prompt = protected_repair_prompt(text, span, policy)
    data = json.loads(prompt.split('SOURCE DATA:\n')[1])
    assert data == {'paragraph': text, 'editable_text': text[span[0]:span[1]]}
    assert 'fiction' in prompt and 'fictional world consistent' in prompt
    assert 'qualification' in prompt and 'causal relationship' in prompt
    with pytest.raises(ValueError):
        protected_repair_prompt(text, (-1, 4), policy)
