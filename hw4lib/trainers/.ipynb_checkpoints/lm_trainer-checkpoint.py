import os
import json
import yaml
import torch
import torch.nn as nn
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple, List
from ..data import H4Tokenizer
from ..decoding.sequence_generator import SequenceGenerator


class LMTrainer:
    """
    Trainer for language model (decoder-only transformer).
    Handles training, validation, evaluation, and text generation.
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer: H4Tokenizer,
        config: Dict[str, Any],
        run_name: str,
        config_file: str,
        device: str = "cuda"
    ):
        """
        Args:
            model: The language model (DecoderOnlyTransformer)
            tokenizer: Tokenizer for encoding/decoding text
            config: Full configuration dictionary
            run_name: Name for this experiment run
            config_file: Path to config.yaml (for saving)
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.run_name = run_name
        self.device = device

        # Create experiment directory
        self.expt_dir = os.path.join("expts", run_name)
        os.makedirs(self.expt_dir, exist_ok=True)
        os.makedirs(os.path.join(self.expt_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.expt_dir, "attn"), exist_ok=True)
        os.makedirs(os.path.join(self.expt_dir, "text"), exist_ok=True)

        # Save config and model architecture
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                saved_config = yaml.safe_load(f)
            with open(os.path.join(self.expt_dir, "config.yaml"), 'w') as f:
                yaml.dump(saved_config, f)

        # Save model architecture summary later (call after model summary is ready)
        self.model_arch_path = os.path.join(self.expt_dir, "model_arch.txt")

        # Loss criterion
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=config['loss']['label_smoothing']
        )

        # Optimizer and scheduler will be set externally
        self.optimizer = None
        self.scheduler = None

        # Wandb logging
        self.use_wandb = config['training']['use_wandb']
        if self.use_wandb:
            wandb.init(
                project=config['training']['wandb_project'],
                name=run_name,
                config=config,
                resume=config['training']['resume'],
                id=config['training']['wandb_run_id'] if config['training']['wandb_run_id'] != "none" else None
            )

    def save_checkpoint(self, checkpoint_type: str = "best"):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.expt_dir, "checkpoints", f"checkpoint-{checkpoint_type}-model.pth"
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Checkpoint loaded: {checkpoint_path}")

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """Log metrics to console and wandb."""
        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.4f}")

        if self.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_perplexity": train_ppl,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            })

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            shifted, golden, lengths = batch
            shifted = shifted.to(self.device)
            golden = golden.to(self.device)
            lengths = lengths.to(self.device)

            logits, _ = self.model(shifted, lengths)  # (B, T, vocab_size)
            loss = self.criterion(logits.view(-1, logits.size(-1)), golden.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                shifted, golden, lengths = batch
                shifted = shifted.to(self.device)
                golden = golden.to(self.device)
                lengths = lengths.to(self.device)

                logits, _ = self.model(shifted, lengths)
                loss = self.criterion(logits.view(-1, logits.size(-1)), golden.view(-1))
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Full training loop."""
        best_val_loss = float('inf')
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_loss = self._validate_epoch(val_loader)

            if self.scheduler is not None and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            self.log_metrics(epoch, train_loss, val_loss)

            # Save checkpoints
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best-metric")
            self.save_checkpoint("last-epoch")

    def generate(
        self,
        prompts: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text using greedy decoding.
        Args:
            prompts: (batch_size, seq_len) token sequences (must include SOS)
            max_length: maximum total length to generate
            temperature: temperature for logits scaling
            repeat_penalty: penalty for repeated tokens
        Returns:
            sequences: (batch_size, generated_length)
            scores: (batch_size,)
        """
        self.model.eval()
        generator = SequenceGenerator(
            score_fn=self.model.score,
            tokenizer=self.tokenizer,
            max_length=max_length,
            device=self.device
        )
        with torch.no_grad():
            sequences, scores = generator.generate_greedy(prompts, temperature, repeat_penalty)
        return sequences, scores

    def evaluate(self, test_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
        """
        Evaluate on test set: compute test loss and character-level perplexity,
        and generate sample text for qualitative evaluation.
        Returns:
            metrics: dict with 'test_loss', 'test_perplexity', 'test_char_perplexity'
            generation_results: dict with 'greedy' list of generated text samples
        """
        self.model.eval()
        total_loss = 0.0
        total_chars = 0
        total_log_prob = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                shifted, golden, lengths = batch
                shifted = shifted.to(self.device)
                golden = golden.to(self.device)
                lengths = lengths.to(self.device)

                logits, _ = self.model(shifted, lengths)  # (B, T, vocab_size)
                loss = self.criterion(logits.view(-1, logits.size(-1)), golden.view(-1))
                total_loss += loss.item()

                # For character-level perplexity: sum log probability of target tokens
                log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(2, golden.unsqueeze(-1)).squeeze(-1)
                # Mask padding positions
                pad_mask = (golden != self.tokenizer.pad_id)
                target_log_probs = target_log_probs * pad_mask
                total_log_prob += target_log_probs.sum().item()
                total_chars += pad_mask.sum().item()

        avg_loss = total_loss / len(test_loader)
        avg_neg_log_prob = -total_log_prob / total_chars if total_chars > 0 else float('inf')
        char_perplexity = np.exp(avg_neg_log_prob)
        token_perplexity = np.exp(avg_loss)

        metrics = {
            "test_loss": avg_loss,
            "test_perplexity": token_perplexity,
            "test_char_perplexity": char_perplexity,
        }

        # Sample prompts for generation
        prompts, originals = test_loader.dataset.sample_prompts(num_samples=4, prompt_length=20, seed=42)
        prompts = prompts.to(self.device)
        generated, _ = self.generate(prompts, max_length=100, temperature=1.0, repeat_penalty=1.0)

        # Decode to text
        generated_texts = []
        for i in range(generated.shape[0]):
            seq = generated[i]
            # Truncate at EOS
            eos_pos = (seq == self.tokenizer.eos_id).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[:eos_pos[0]+1]
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            generated_texts.append(text)

        # Also decode original prompts for reference
        original_texts = [self.tokenizer.decode(orig.tolist(), skip_special_tokens=True) for orig in originals]

        generation_results = {
            "greedy": {
                "prompts": original_texts,
                "generated": generated_texts,
            }
        }

        # Save results
        with open(os.path.join(self.expt_dir, "test_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(self.expt_dir, "test_generated_results.json"), 'w') as f:
            json.dump(generation_results, f, indent=4)

        print(f"Test Loss: {avg_loss:.4f}, Token PPL: {token_perplexity:.4f}, Char PPL: {char_perplexity:.4f}")
        return metrics, generation_results

    def cleanup(self):
        """Close wandb run."""
        if self.use_wandb:
            wandb.finish()