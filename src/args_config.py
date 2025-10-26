# src/args_config.py
import argparse
import os
import torch
import random
import numpy as np
import logging

class AHPSettings:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Run AHP and SelfDenoise Experiments")

        # === Mode ===
        self.parser.add_argument('--mode', type=str, default='attack', choices=['attack', 'evaluate'],
                                 help="Operation mode: 'attack' or 'evaluate' clean accuracy.")

        # === Model ===
        self.parser.add_argument('--model_path', type=str, default='/root/autodl-tmp/circulus_alpaca-7b',
                                 help="Path to the pre-trained Alpaca model.")
        self.parser.add_argument('--cache_dir', type=str, default='./cache_path',
                                 help="Directory for caching models and datasets.")
        self.parser.add_argument('--model_batch_size', type=int, default=4,
                                 help="Batch size for model inference.")

        # === Dataset ===
        self.parser.add_argument('--dataset_name', type=str, default='sst2', choices=['sst2', 'agnews'],
                                 help="Dataset to use: 'sst2' or 'agnews'.")
        self.parser.add_argument('--dataset_path', type=str, default='./data', # 假设您的数据在此
                                 help="Base path to the dataset directory.")
        self.parser.add_argument('--num_examples', type=int, default=100,
                                 help="Number of examples from the test set to use for evaluation/attack.")
        self.parser.add_argument('--max_seq_length', type=int, default=128,
                                 help="Maximum sequence length for tokenization.")


        # === Defense ===
        self.parser.add_argument('--defense_method', type=str, default='none', choices=['none', 'ahp', 'selfdenoise'],
                                 help="Defense method to apply: 'none', 'ahp', or 'selfdenoise'.")
        self.parser.add_argument('--mask_token', type=str, default='<MASK>',
                                 help="Token used for masking.")
        self.parser.add_argument('--mask_rate', type=float, default=0.15,
                                 help="Masking rate for AHP and SelfDenoise.")

        # --- AHP Specific ---
        self.parser.add_argument('--ahp_num_candidates', type=int, default=10,
                                 help="Number of candidates to generate per masked input in AHP.")
        self.parser.add_argument('--ahp_pruning_method', type=str, default='perplexity', choices=['perplexity', 'semantic', 'nli', 'clustering', 'none'],
                                 help="Pruning method for AHP candidates.")
        self.parser.add_argument('--ahp_pruning_threshold', type=float, default=0.7,
                                 help="Threshold for pruning (meaning depends on method).")
        self.parser.add_argument('--ahp_aggregation_strategy', type=str, default='majority_vote', choices=['majority_vote', 'weighted_vote'],
                                 help="Aggregation strategy for AHP.")

        # --- SelfDenoise Specific ---
        self.parser.add_argument('--selfdenoise_ensemble_size', type=int, default=50,
                                 help="Number of masked samples for SelfDenoise prediction ensemble.")
        # self.parser.add_argument('--selfdenoise_certify_ensemble', type=int, default=1000,
        #                          help="Number of masked samples for SelfDenoise certification (larger).") # Certify mode暂不实现
        self.parser.add_argument('--selfdenoise_denoiser', type=str, default='alpaca', choices=['alpaca', 'roberta'],
                                 help="Model to use for denoising in SelfDenoise.")

        # === Attack ===
        self.parser.add_argument('--attack_method', type=str, default='textbugger', choices=['textbugger', 'textfooler', 'pwws', 'bae', 'deepwordbug'],
                                 help="TextAttack recipe to use.")
        self.parser.add_argument('--attack_query_budget', type=int, default=100,
                                 help="Maximum number of model queries allowed per attack.")
        self.parser.add_argument('--attack_log_path', type=str, default='./results/attack_logs',
                                 help="Directory to save detailed attack logs.")
        self.parser.add_argument('--results_file', type=str, default='./results/experiment_results.csv',
                                 help="CSV file to append overall results.")


        # === Environment ===
        self.parser.add_argument('--seed', type=int, default=42,
                                 help="Random seed for reproducibility.")
        self.parser.add_argument('--device', type=str, default=None,
                                 help="Device to use ('cuda', 'cpu'). Auto-detects if None.")
        self.parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                 help="Logging level.")


    def parse_args(self, args_list=None):
        args = self.parser.parse_args(args_list) # Provide args_list for testing/notebook usage

        # --- Post-processing and Setup ---
        # Setup logging
        logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Setup device
        if args.device is None:
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {args.device}")

        # Set seed
        self.set_random_seed(args.seed, deterministic=False) # Keep deterministic=False for speed unless needed
        logging.info(f"Set random seed to: {args.seed}")

        # Create output dirs
        os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
        os.makedirs(args.attack_log_path, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

        # Dataset full path
        args.dataset_full_path = os.path.join(args.dataset_path, args.dataset_name)

        return args

    @staticmethod
    def set_random_seed(seed, deterministic=False):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            logging.warning("Using deterministic algorithms, may be slower.")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # For newer PyTorch versions:
            # torch.use_deterministic_algorithms(True)