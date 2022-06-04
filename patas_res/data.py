from collections import Counter
from typing import Any, Callable

import numpy as np
from hw6_data import pad_batch
from vocabulary import Vocabulary

# type aliases
Example = dict[str, Any]


class Dataset:
    def __init__(self, examples: list[Example], vocab: Vocabulary) -> None:
        self.examples = examples
        self.vocab = vocab
        self.num_labels = len(self.vocab)
        self._label_one_hots = np.eye(self.num_labels)

    def example_to_tensors(self, index: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def batch_as_tensors(self, start: int, end: int) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]

    def __len__(self) -> int:
        return len(self.examples)


def example_from_characters(source: list[str], target: list[str], bos: str, eos: str) -> Example:
    return {
        'source': [bos] + source,
        'target_x': [bos] + target,
        'target_y': target + [eos],
        'lengths': (len(source) + 1, len(target) + 1)
    }


class Seq2SeqDataset(Dataset):
    """
    A dataset for sequence to sequence pairs that inherits from Dataset class. Contains examples, a vocabulary, 
    number of labels, and one hots for the labels. Examples can be read in from files. Each example is indexed and 
    can be obtained by passing in the index to the example_to_indices function. Examples can also be batched over 
    index ranges.
    """
    BOS = "<s>"
    EOS = "</s>"
    PAD = "<PAD>"

    def example_to_indices(self, index: int) -> dict[str, Any]:
        """
        Takes an index and returns
        Args:
            index:

        Returns:

        """
        example = self.__getitem__(index)
        return {
            'source': np.array(self.vocab.tokens_to_indices(example['source'])),
            'target_x': np.array(self.vocab.tokens_to_indices(example['target_x'])),
            'target_y': self.vocab.tokens_to_indices(example['target_y']),
            'lengths': example['lengths']
        }

    def batch_as_tensors(self, start: int, end: int) -> dict[str, Any]:
        """
        This method pulls examples within a given index range and adds padding to the examples. Collects and returns
        a dictionary with keys into various features of the examples.
        Args:
            start: lower bound index of the range of examples to pull
            end: upper bound index of the the range of examples to pull.
        Returns: a dictionary with 4 key-value pairs: the source is padded text used as input to the encoder;
         the target x is the input sequence of tokens to the decoder,
         the target y is the output sequence from the decoder.
         the lengths is a tuple of the lengths of source sequence and target sequence.
        """
        examples = [self.example_to_indices(index) for index in range(start, end)]
        padding_index = self.vocab[Seq2SeqDataset.PAD]
        # pad texts to [batch_size, max_seq_len] array
        source = pad_batch([example['source'] for example in examples], padding_index)
        source_lengths = [example['lengths'][0] for example in examples]
        # target: [batch_size, max_seq_len], indices for next character
        target_x = pad_batch([example['target_x'] for example in examples], padding_index)
        target_y = pad_batch([example['target_y'] for example in examples], padding_index)
        target_lengths = [example['lengths'][1] for example in examples]
        return {
            'source': source,
            'target_x': target_x,
            'target_y': target_y,
            'lengths': (source_lengths, target_lengths)
        }

    @classmethod
    def from_files(cls, source_file: str, target_file: str, vocab: Vocabulary = None):
        """
        Use sequences from source and target file to create examples for the seq2seq dataset. Creates vocabulary from
         examples if it doesn't exist yet.
        Args:
            source_file: the source file for example creation
            target_file: the target file for example creation
            vocab: the vocabulary to pass in. Defaults to None.

        Returns: examples and vocab, setting those variables for the Seq2SeqDataset class state.

        """
        examples = []
        counter: Counter = Counter()
        source_lines = [line.strip('\n').lower() for line in open(source_file, 'r')]
        target_lines = [line.strip('\n').lower() for line in open(target_file, 'r')]
        lines = list(zip(source_lines, target_lines))
        for line in lines:
            counter.update(list(line[0]))
            counter.update(list(line[1]))
            # generate example from line
            example = example_from_characters(
                list(line[0]),
                list(line[1]),
                Seq2SeqDataset.BOS,
                Seq2SeqDataset.EOS
            )
            examples.append(example)
        if not vocab:
            vocab = Vocabulary(
                counter,
                special_tokens=(
                    Vocabulary.UNK,
                    Seq2SeqDataset.BOS,
                    Seq2SeqDataset.EOS,
                    Seq2SeqDataset.PAD,
                ),
            )
        return cls(examples, vocab)
