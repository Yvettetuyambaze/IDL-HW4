from typing import Literal, Tuple, Optional
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as tat
import re
from .tokenizer import H4Tokenizer

'''
TODO: Implement this class.

Specification:
The ASRDataset class provides data loading and processing for ASR (Automatic Speech Recognition):

1. Data Organization:
   - Handles dataset partitions (train-clean-100, dev-clean, test-clean)
   - Features stored as .npy files in fbank directory
   - Transcripts stored as .npy files in text directory
   - Maintains alignment between features and transcripts

2. Feature Processing:
   - Loads log mel filterbank features from .npy files
   - Supports multiple normalization strategies:
     * global_mvn: Global mean and variance normalization
     * cepstral: Per-utterance mean and variance normalization
     * none: No normalization
   - Applies SpecAugment data augmentation during training:
     * Time masking: Masks random time steps
     * Frequency masking: Masks random frequency bands

3. Transcript Processing:
   - Similar to LMDataset transcript handling
   - Creates shifted (SOS-prefixed) and golden (EOS-suffixed) versions
   - Tracks statistics for perplexity calculation
   - Handles tokenization using H4Tokenizer

4. Batch Preparation:
   - Pads features and transcripts to batch-uniform lengths
   - Provides lengths for packed sequence processing
   - Ensures proper device placement and tensor types

Key Requirements:
- Must maintain feature-transcript alignment
- Must handle variable-length sequences
- Must track maximum lengths for both features and text
- Must implement proper padding for batching
- Must apply SpecAugment only during training
- Must support different normalization strategies
'''

class ASRDataset(Dataset):
    def __init__(
            self,
            partition:Literal['train-clean-100', 'dev-clean', 'test-clean'],
            config:dict,
            tokenizer:H4Tokenizer,
            isTrainPartition:bool,
            global_stats:Optional[Tuple[torch.Tensor, torch.Tensor]]=None
    ):
        """
        Initialize the ASRDataset for ASR training/validation/testing.
        Args:
            partition (str): Dataset partition ('train-clean-100', 'dev-clean', or 'test-clean')
            config (dict): Configuration dictionary containing dataset settings
            tokenizer (H4Tokenizer): Tokenizer for encoding/decoding text
            isTrainPartition (bool): Whether this is the training partition
                                     Used to determine if SpecAugment should be applied.
            global_stats (tuple, optional): (mean, std) computed from training set.
                                          If None and using global_mvn, will compute during loading.
                                          Should only be None for training set.
                                          Should be provided for dev and test sets.
        """
        # Store basic configuration
        self.config    = config
        self.partition = partition
        self.isTrainPartition = isTrainPartition
        self.tokenizer = tokenizer

        # Get tokenizer ids for special tokens (eos, sos, pad)
        self.eos_token = tokenizer.eos_id
        self.sos_token = tokenizer.sos_id
        self.pad_token = tokenizer.pad_id

        # Set up data paths 
        # Use root and partition to get the feature directory
        self.fbank_dir = os.path.join(config['root'], partition, 'fbank')
        
        # Get all feature files in the feature directory in sorted order  
        self.fbank_files = sorted([os.path.join(self.fbank_dir, f) for f in os.listdir(self.fbank_dir) if f.endswith('.npy')])
        
        # Take subset
        subset_size = int(len(self.fbank_files) * config.get('subset', 1.0))
        self.fbank_files = self.fbank_files[:subset_size]
        
        # Get the number of samples in the dataset  
        self.length = len(self.fbank_files)

        # Case on partition.
        # Why will test-clean need to be handled differently?
        if self.partition != "test-clean":
            # Use root and partition to get the text directory
            self.text_dir = os.path.join(config['root'], partition, 'text')

            # Derive text_files directly from fbank_files by replacing the directory.
            # This guarantees perfect 1-to-1 alignment: each fbank file maps to the
            # transcript file with the identical basename in the text directory.
            self.text_files = [
                os.path.join(self.text_dir, os.path.basename(f))
                for f in self.fbank_files
            ]

            # Verify data alignment: lengths must match
            if len(self.fbank_files) != len(self.text_files):
                raise ValueError("Number of feature and transcript files must match")
        else:
            self.text_files = None

        # Initialize lists to store features and transcripts
        self.feats, self.transcripts_shifted, self.transcripts_golden = [], [], []
        
        # Initialize counters for character and token counts
        # DO NOT MODIFY
        self.total_chars  = 0
        self.total_tokens = 0
        
        # Initialize max length variables
        # DO NOT MODIFY
        self.feat_max_len = 0
        self.text_max_len = 0
        
        # Initialize Welford's algorithm accumulators if needed for global_mvn
        # DO NOT MODIFY
        if self.config['norm'] == 'global_mvn' and global_stats is None:
            if not isTrainPartition:
                raise ValueError("global_stats must be provided for non-training partitions when using global_mvn")
            count = 0
            mean = torch.zeros(self.config['num_feats'], dtype=torch.float64)
            M2 = torch.zeros(self.config['num_feats'], dtype=torch.float64)

        print(f"Loading data for {partition} partition...")
        for i in tqdm(range(self.length)):
            # Load features
            # Features are of shape (num_feats, time)
            feat = np.load(self.fbank_files[i]).astype(np.float32)

            # Truncate features to num_feats set by you in the config
            if feat.shape[0] != self.config['num_feats']:
                feat = feat[:self.config['num_feats'], :]

            # Append to self.feats (num_feats is set by you in the config)
            self.feats.append(feat)

            # Track max length (time dimension)
            self.feat_max_len = max(self.feat_max_len, feat.shape[1])

            # Update global statistics if needed (DO NOT MODIFY)
            if self.config['norm'] == 'global_mvn' and global_stats is None:
                feat_tensor = torch.FloatTensor(feat)  # (num_feats, time)
                batch_count = feat_tensor.shape[1]     # number of time steps
                count += batch_count
                
                # Update mean and M2 for all time steps at once
                delta = feat_tensor - mean.unsqueeze(1)  # (num_feats, time)
                mean += delta.mean(dim=1)                # (num_feats,)
                delta2 = feat_tensor - mean.unsqueeze(1) # (num_feats, time)
                M2 += (delta * delta2).sum(dim=1)        # (num_feats,)

            # NOTE: The following steps are almost the same as the steps in the LMDataset   
            
            if self.partition != "test-clean":
                # Load the transcript
                # Important Note: This is a very important line of code and you should check whether your transcript is correct after loading (and very dependent in evaluation)
                transcript_data = np.load(self.text_files[i], allow_pickle=True)
                # Robust conversion to string
                if isinstance(transcript_data, np.ndarray):
                    if transcript_data.size == 1:
                        transcript = transcript_data.item()
                        if isinstance(transcript, bytes):
                            transcript = transcript.decode('utf-8')
                    else:
                        # Assume array of characters
                        transcript = ''.join([str(x) for x in transcript_data])
                else:
                    transcript = str(transcript_data)

                # Strip the LibriSpeech utterance ID prefix if present.
                # Text .npy files often store the full line as:
                #   "103-1240-0000 I WAS THE YOUNGEST"
                # We only want the spoken text portion, not the utterance ID.
                # The utterance ID has the form SPEAKER-CHAPTER-UTTERANCE (digits and dashes)
                # followed by a single space before the transcript text.
                transcript = re.sub(r'^\d+-\d+-\d+\s+', '', transcript)

                # Track character count (before tokenization)
                self.total_chars += len(transcript)

                # Use tokenizer to encode the transcript (see tokenizer.encode for details)
                tokenized = tokenizer.encode(transcript)

                # Track token count (excluding special tokens)
                # DO NOT MODIFY
                self.total_tokens += len(tokenized)

                # Track max length (add 1 for the sos/eos tokens)
                # DO NOT MODIFY
                self.text_max_len = max(self.text_max_len, len(tokenized)+1)
                
                # Create shifted and golden versions by adding sos and eos tokens
                shifted = [self.sos_token] + tokenized
                golden = tokenized + [self.eos_token]
                self.transcripts_shifted.append(torch.tensor(shifted, dtype=torch.long))
                self.transcripts_golden.append(torch.tensor(golden, dtype=torch.long))

        # Calculate average characters per token
        # DO NOT MODIFY 
        self.avg_chars_per_token = self.total_chars / self.total_tokens if self.total_tokens > 0 else 0
        
        if self.partition != "test-clean":
            # Verify data alignment
            if not (len(self.feats) == len(self.transcripts_shifted) == len(self.transcripts_golden)):
                raise ValueError("Features and transcripts are misaligned")

        # Compute final global statistics if needed
        if self.config['norm'] == 'global_mvn':
            if global_stats is not None:
                self.global_mean, self.global_std = global_stats
            else:
                # Compute variance and standard deviation
                variance = M2/(count - 1)
                self.global_std = torch.sqrt(variance + 1e-8).float()
                self.global_mean = mean.float()

        # Initialize SpecAugment transforms
        self.time_mask = tat.TimeMasking(
            time_mask_param=config['specaug_conf']['time_mask_width_range'],
            iid_masks=True
        )
        self.freq_mask = tat.FrequencyMasking(
            freq_mask_param=config['specaug_conf']['freq_mask_width_range'],
            iid_masks=True
        )

    def get_avg_chars_per_token(self):
        '''
        Get the average number of characters per token. Used to calculate character-level perplexity.
        DO NOT MODIFY
        '''
        return self.avg_chars_per_token

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        DO NOT MODIFY
        """
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (features, shifted_transcript, golden_transcript) where:
                - features: FloatTensor of shape (num_feats, time)
                - shifted_transcript: LongTensor (time) or None
                - golden_transcript: LongTensor  (time) or None
        """
        # Load features
        feat = self.feats[idx].copy()
        feat = torch.from_numpy(feat).float()

        # Apply normalization
        if self.config['norm'] == 'global_mvn':
            assert hasattr(self, 'global_mean') and hasattr(self, 'global_std'), "Global mean and std must be computed before normalization"
            feat = (feat - self.global_mean.unsqueeze(1)) / (self.global_std.unsqueeze(1) + 1e-8)
        elif self.config['norm'] == 'cepstral':
            feat = (feat - feat.mean(dim=1, keepdim=True)) / (feat.std(dim=1, keepdim=True) + 1e-8)
        elif self.config['norm'] == 'none':
            pass
        
        # Get transcripts for non-test partitions
        shifted_transcript, golden_transcript = None, None
        if self.partition != "test-clean":
            shifted_transcript = self.transcripts_shifted[idx]
            golden_transcript = self.transcripts_golden[idx]

        return feat, shifted_transcript, golden_transcript

    def collate_fn(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate and pad a batch of samples to create a batch of fixed-length padded features and transcripts.

        Args:
            batch (list): List of samples from __getitem__

        Returns:
            tuple: (padded_features, padded_shifted, padded_golden, feat_lengths, transcript_lengths) where:
                - padded_features: Tensor of shape (batch, max_time, num_feats)
                - padded_shifted: Tensor of shape (batch, max_len) or None
                - padded_golden: Tensor of shape (batch, max_len) or None  
                - feat_lengths: Tensor of original feature lengths of shape (batch)
                - transcript_lengths: Tensor of transcript lengths of shape (batch) or None
        """
        # Collect transposed features from the batch into a list of tensors (B x T x F)
        batch_feats = []
        for sample in batch:
            feat, _, _ = sample
            # Transpose from (C, T) to (T, C)
            batch_feats.append(feat.t())

        # Collect feature lengths from the batch into a tensor
        feat_lengths = torch.tensor([f.size(0) for f in batch_feats], dtype=torch.long)

        # Pad features to create a batch of fixed-length padded features
        padded_feats = pad_sequence(batch_feats, batch_first=True, padding_value=0.0)  # (B, max_T, C)

        # Handle transcripts for non-test partitions
        padded_shifted, padded_golden, transcript_lengths = None, None, None
        if self.partition != "test-clean":
            # Collect shifted and golden transcripts from the batch into a list of tensors (B x T)  
            batch_shifted = []
            batch_golden = []
            for sample in batch:
                _, shifted, golden = sample
                batch_shifted.append(shifted)
                batch_golden.append(golden)

            # Collect transcript lengths from the batch into a tensor
            transcript_lengths = torch.tensor([len(t) for t in batch_shifted], dtype=torch.long)

            # Pad transcripts to create a batch of fixed-length padded transcripts
            padded_shifted = pad_sequence(batch_shifted, batch_first=True, padding_value=self.pad_token)
            padded_golden = pad_sequence(batch_golden, batch_first=True, padding_value=self.pad_token)

        # Apply SpecAugment for training
        if self.config["specaug"] and self.isTrainPartition:
            # Permute the features to (B x F x T)
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, C, T)

            # Apply frequency masking
            if self.config["specaug_conf"]["apply_freq_mask"]:
                for _ in range(self.config["specaug_conf"]["num_freq_mask"]):
                    padded_feats = self.freq_mask(padded_feats)

            # Apply time masking
            if self.config["specaug_conf"]["apply_time_mask"]:
                for _ in range(self.config["specaug_conf"]["num_time_mask"]):
                    padded_feats = self.time_mask(padded_feats)

            # Permute the features back to (B x T x F)
            padded_feats = padded_feats.permute(0, 2, 1)  # (B, T, C)

        # Return the padded features, padded shifted, padded golden, feature lengths, and transcript lengths
        return padded_feats, padded_shifted, padded_golden, feat_lengths, transcript_lengths