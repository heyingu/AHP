# run_experiment.py
import logging
import sys
# Add src to Python path if run from root directory
sys.path.insert(0, './src')

from src.args_config import AHPSettings
from src.experiment_runner import ExperimentRunner

def main():
    # 1. Parse Arguments
    args = AHPSettings().parse_args()
    logging.info("Starting experiment with arguments:")
    for key, value in vars(args).items():
        logging.info(f"  {key}: {value}")

    # 2. Initialize Runner
    runner = ExperimentRunner(args)

    # 3. Run Experiment
    runner.run()

    logging.info("Experiment finished.")

if __name__ == "__main__":
    main()