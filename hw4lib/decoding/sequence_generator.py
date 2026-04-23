import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        batch_size, seq_len = x.shape
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - seq_len):
            if finished.all():
                break
            logits = self.score_fn(x)
            if repeat_penalty != 1.0:
                logits = self._apply_repeat_penalty(logits, x, repeat_penalty)
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)
            next_tokens = torch.argmax(log_probs, dim=-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores = torch.where(finished, scores, scores + token_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | (next_tokens == self.tokenizer.eos_id)
        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
             - scores is of shape (batch_size, beam_width)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        batch_size, seq_len = x.shape
        device = x.device

        # Process each batch item independently to avoid flattening issues
        all_seqs = []
        all_scores = []

        for b in range(batch_size):
            # Initial sequence for this batch item
            cur_x = x[b:b+1]                     # (1, seq_len)
            # Expand to beam_width copies
            beams = cur_x.repeat(beam_width, 1)   # (beam_width, seq_len)
            scores = torch.zeros(beam_width, device=device)
            finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

            # First step
            logits = self.score_fn(cur_x)         # (1, vocab_size)
            if repeat_penalty != 1.0:
                logits = self._apply_repeat_penalty(logits, cur_x, repeat_penalty)
            logits = logits / temperature
            log_probs = torch.log_softmax(logits, dim=-1)   # (1, vocab_size)
            top_log_probs, top_tokens = torch.topk(log_probs[0], beam_width)
            scores = top_log_probs
            beams = torch.cat([beams, top_tokens.unsqueeze(1)], dim=1)
            finished = finished | (top_tokens == self.tokenizer.eos_id)

            # Remaining steps
            for _ in range(self.max_length - seq_len - 1):
                if finished.all():
                    break
                # For each beam, compute logits (call score_fn with (1, L))
                all_candidates = []
                for k in range(beam_width):
                    prev_seq = beams[k]
                    if finished[k]:
                        # Preserve finished beams as candidates with unchanged score
                        all_candidates.append((k, None, scores[k].item(), True))
                        continue
                    seq = prev_seq.unsqueeze(0)            # (1, L)
                    logits = self.score_fn(seq)            # (1, vocab_size)
                    if repeat_penalty != 1.0:
                        logits = self._apply_repeat_penalty(logits, seq, repeat_penalty)
                    logits = logits / temperature
                    log_probs = torch.log_softmax(logits, dim=-1)  # (1, vocab_size)
                    cum_log_probs = scores[k] + log_probs[0]       # (vocab_size,)
                    for token_id, score_val in enumerate(cum_log_probs):
                        all_candidates.append((k, token_id, score_val.item(), False))
                if not all_candidates:
                    break
                # Flatten all candidates and select the top beams
                all_candidates.sort(key=lambda x: x[2], reverse=True)
                top_candidates = all_candidates[:beam_width]
                # Build new beams from selected candidates
                new_beams = []
                new_scores = []
                new_finished = []
                for k, token_id, score_val, is_finished in top_candidates:
                    prev_seq = beams[k]
                    if is_finished:
                        new_seq = prev_seq
                    else:
                        new_seq = torch.cat([prev_seq, torch.tensor([token_id], device=device)])
                    new_beams.append(new_seq)
                    new_scores.append(score_val)
                    new_finished.append(is_finished or finished[k] or (token_id == self.tokenizer.eos_id))
                beams = torch.stack(new_beams)
                scores = torch.tensor(new_scores, device=device)
                finished = torch.tensor(new_finished, dtype=torch.bool, device=device)

            # Sort beams by score descending
            sort_idx = torch.argsort(scores, descending=True)
            beams = beams[sort_idx]
            scores = scores[sort_idx]
            all_seqs.append(beams)
            all_scores.append(scores)

        # Pad sequences to same length
        # all_seqs is a list (over batch) of tensors of shape (beam_width, seq_len).
        # Iterating over a 2-D tensor yields 1-D rows, so use .size(0) not .size(1).
        max_len = max(seq.size(0) for seqs in all_seqs for seq in seqs)
        padded_seqs = []
        for batch_seqs in all_seqs:
            batch_padded = []
            for seq in batch_seqs:
                if seq.size(0) < max_len:
                    pad = torch.full((max_len - seq.size(0),), self.tokenizer.pad_id, device=device)
                    seq = torch.cat([seq, pad])
                batch_padded.append(seq)
            padded_seqs.append(torch.stack(batch_padded))
        sequences = torch.stack(padded_seqs)   # (batch_size, beam_width, max_len)
        scores = torch.stack(all_scores)       # (batch_size, beam_width)
        return sequences, scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            if finished.all():
                break
            next_scores = self.score_fn(x)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)
            scores = torch.where(finished, scores, scores + token_scores)
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | (next_tokens == self.tokenizer.eos_id)
        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        eos_mask = seq == tokenizer.eos_id
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]