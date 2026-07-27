# core/auto_trainer.py

import time
import yaml
from storage.database import count_unused_pairs, get_state
from core.training_engine import TrainingEngine

def load_config():
    with open("config/aion_config.yaml") as f:
        return yaml.safe_load(f)

def run_auto_trainer():
    print("[AutoTrainer] Started background daemon...")
    config = load_config()
    interval = config["training"]["check_interval_seconds"]
    min_new_questions = config["training"]["min_new_questions"] 
    
    engine = TrainingEngine()
    
    while True:
        try:
            time.sleep(interval)
            
            if get_state("model_frozen") == "true":
                continue
                
            unused = count_unused_pairs()
            if unused >= min_new_questions:
                print(f"[AutoTrainer] Found {unused} new training pairs. Triggering training.")
                engine.train()
                
        except Exception as e:
            print(f"[AutoTrainer] Error in background loop: {e}")

if __name__ == "__main__":
    run_auto_trainer()
