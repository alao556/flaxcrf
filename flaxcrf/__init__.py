__version__ = '0.1.0'
# This file is modified from https://github.com/yumoh/torchcrf/blob/master/torchcrf/__init__.py

from typing import List, Optional

import jax.numpy as jnp
import flax.linen as nn
from flax.core.scope import VariableDict
from flax.linen.initializers import uniform


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Learnable parameter:
        start_transitions: Start transition score tensor of size ``(num_tags,)``.
        end_transitions: End transition score tensor of size ``(num_tags,)``.
        transitions: Transition score tensor of size ``(num_tags, num_tags)``.

    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    num_tags: int
    batch_first: bool = False

    def setup(self) -> None:
        if self.num_tags <= 0:
            raise ValueError(f"invalid number of tags: {self.num_tags}")
        self.start_transitions = self.param("start_transitions", uniform(), [self.num_tags])
        self.end_transitions = self.param("end_transitions", uniform(), [self.num_tags])
        self.transitions = self.param("transitions", uniform(), [self.num_tags, self.num_tags])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def __call__(
            self,
            emissions: jnp.ndarray,
            tags: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
            reduction: str = "sum",
    ) -> jnp.ndarray:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~jnp.ndarray`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~jnp.ndarray`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~jnp.ndarray`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~jnp.ndarray`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = jnp.ones_like(tags, dtype=jnp.uint8)

        if self.batch_first:
            emissions = emissions.swapaxes(0, 1)
            tags = tags.swapaxes(0, 1)
            mask = mask.swapaxes(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        assert reduction == "token_mean"
        return llh.sum() / mask.astype("float32").sum()

    def decode(
            self,
            params: VariableDict,
            emissions: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None
    ) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            params: ...
            emissions (`~jnp.ndarray`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~jnp.ndarray`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = jnp.ones(emissions.shape[:2], dtype=jnp.uint8)

        if self.batch_first:
            emissions = emissions.swapaxes(0, 1)
            mask = mask.swapaxes(0, 1)

        return self._viterbi_decode(params, emissions, mask)

    def _validate(
            self,
            emissions: jnp.ndarray,
            tags: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None) -> None:
        if emissions.ndim != 3:
            raise ValueError(f"emissions must have dimension of 3, got {emissions.ndim}")
        if emissions.shape[2] != self.num_tags:
            raise ValueError(f"expected last dimension of emissions is {self.num_tags}, got {emissions.shape[2]}")

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    "the first two dimensions of emissions and tags must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}")

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}")
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _compute_score(self, emissions: jnp.ndarray, tags: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.ndim == 3 and tags.ndim == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.shape[2] == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.astype("float32")

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, jnp.arange(0, batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, jnp.arange(0, batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.astype("int32").sum(axis=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, jnp.arange(0, batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.ndim == 3 and mask.ndim == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask[0].all()

        seq_length = emissions.shape[0]

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = jnp.expand_dims(score, axis=2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = jnp.expand_dims(emissions[i], axis=1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = nn.logsumexp(next_score, axis=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = jnp.where(jnp.expand_dims(mask[i], axis=1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return nn.logsumexp(score, axis=1)

    def _viterbi_decode(self, variables: VariableDict, emissions: jnp.ndarray, mask: jnp.ndarray) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.ndim == 3 and mask.ndim == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        start_transitions = variables["params"]["start_transitions"]
        score = start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = jnp.expand_dims(score, axis=2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = jnp.expand_dims(emissions[i], axis=1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            transitions = variables["params"]["transitions"]
            next_score = broadcast_score + transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            indices = next_score.argmax(axis=1)
            next_score = next_score.max(axis=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)

            score = jnp.where(jnp.expand_dims(mask[i], axis=1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        end_transitions = variables["params"]["end_transitions"]
        score += end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.astype("int32").sum(axis=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            best_last_tag = score[idx].argmax(axis=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
