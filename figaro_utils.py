import sys

sys.path.append('figaro/src')
from datasets import MidiDataset


def get_description_from_midi_path(path):
    ds = MidiDataset(
        [path],
        max_len=-1
    )

    for item in ds:
        result = item['description']

    return result
