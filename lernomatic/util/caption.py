"""
CAPTION
Utilities for generating captions

Stefan Wong 2019
"""

# TODO : the entire codebase ought to be re-implemented with type hints
import torch
import heapq

# Top-N elements of an incrementally provided set
class CaptionTopN(object):
    def __init__(self, n) -> None:
        self.n = n
        self.data = []

    def reset(self):
        self.data = []

    def size(self) -> int:
        if self.data is not None:
            return len(self.data)
        return 0

    def push(self, X) -> None:
        if self.data is None:
            return
        if len(self.data) < self.n:
            heapq.heappush(self.data, X)
        else:
            heapq.heappush(self.data, X)

    def extract(self, sort=False) -> list:
        """
        Extact all elements from CaptionTopN. This is a destructive
        operation.

        Args:
            sort : whether to return elements in descending order

        Outputs:
            A list containing the top-N elements provided to the object
        """
        if self.data is None:
            return []


# a partial or complete caption
class Caption(object):
    def __init__(self, sentence, model_state, logprob, score, meta=None) -> None:
        self.sentence    = sentence
        self.model_state = model_state
        self.logprob     = logprob
        self.score       = score
        self.meta        = meta

    def __repr__(self) -> str:
        return 'Caption'

    def __str__(self) -> str:
        return 'Caption [%s] ' % str(self.sentence)

    def __eq__(self, other : 'Caption') -> bool:
        if not isinstance(other, Caption):
            return False
        return self.score == other.score

    def __lt__(self, other : 'Caption') -> bool:
        if not isinstance(other, Caption):
            return False
        return self.score < other.score

    def __gt__(self, other : 'Caption') -> bool:
        if not isinstance(other, Caption):
            return False
        return self.score > other.score


class CaptionGen(object):
    """
    Finds captions using beam search
    """
    def __init__(self, embedder, rnn, classifier, **kwargs) -> None:
        self.beam_size   : int = kwargs.pop('beam_size', 3)
        self.eos_id      : str = kwargs.pop('eos_id', '<EOS>')
        self.max_cap_len : int = kwargs.pop('max_cap_len', 40)

        # Models
        # TODO : I am more or less copying this from a Tensorflow example, but
        # I don't like the way that its structured. Need to re-organise this
        # object before merging into master. Specifically, I'm on the fence
        # about having model references here
        self.embedder   = embedder
        self.rnn        = rnn
        self.classifier = classifier

    def get_topk_words(self, embeddings, state):
        output, new_states = self.rnn(embeddings, state)
        output = self.classifier(output.squeeze(0))
        logprobs = torch.nn.functional.log_softmax(output)
        logprobs, words = logprobs.topk(self.beam_size, 1)

        return (words.item(), logprobs.item(), new_states)

    def beam_search(self, X, initial_state=None):


        partial_captions = CaptionTopN(self.beam_size)
        complete_captions = CaptionTopN(self.beam_size)

        words, logprobs, new_state = self.get_topk_words(X, initial_state)
        for k in range(self.beam_size):
            cap = Caption(
                sentence = [words[0, k]],
                state = new_state,
                logprob = logprobs[0, k],
                score = logprobs[0, k]
            )
            partial_captions.push(cap)

        # run beam search
        for n in range(self.max_cap_len-1):
            partial_captions_lsit = partial_captions.extract()
            partial_captions.reset()

            input_feed = torch.LongTensor([c.sentence[-1] for c in partial_captions_list])






