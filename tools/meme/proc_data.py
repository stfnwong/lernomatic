"""
Process meme json


"""
import json
from typing import Any, Dict, List

#from lernomatic.data.word_map import WordMap


def process_json(filename:str, **kwargs) -> Dict[str, Any]:

    max_rows:int = kwargs.pop('max_rows', 2000000)

    with open(filename, 'r') as fp:
        data = json.load(fp)

    text_samples = []          # collated sets of text samples
    label_map = dict()
    next_label_id = 0
    labels:List[int]= []         # label_id list

    for n, row in enumerate(data):
        template_id = str(row[0]).zfill(12)
        text = row[1].lower()

        # format here is <template_id>, <spaces>, <box_idx>, <spaces>
        start_idx = len(template_id) + 2 + 1 + 2
        box_idx = 0

        for tt in range(0, len(text)):
            cur_char = text[tt]
            # we want to ensure that the number of spaces + len(box_idx) is >=
            # the convolution width in the network
            text_samples.append(template_id + ' ' + str(box_idx) + ' ' + text[0:j])
            if cur_char in label_map:
                label_id = labal_map[cur_char]
            else:
                label_id = next_label_id
                label_map[cur_char] = label_id
                next_label_id += 1

            labels.append(label_id)

            if char == '|':     # box delimiter char
                box_idx += 1

        if n >= max_rows:
            break

    return {
        'text_samples': text_samples,
        'label_map': label_map
    }


def map_char_to_int(text_samples:List[str]) -> Dict[str, int]:
    pass

def text_to_sequence(text_samples:List[str], char_map:Dict[str, int]) -> List[int]:
    pass


# TODO: test, remove this
if __name__ == '__main__':
    FILENAME = "data/meme_test_data.json"

    process_json(FILENAME)
