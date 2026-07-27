import os
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class DatasetWatchdogHandler(FileSystemEventHandler):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.last_trigger = 0
        self.cooldown = 5  # debounce trigger in seconds

    def on_created(self, event):
        self._handle_event(event)
        
    def on_modified(self, event):
        self._handle_event(event)
        
    def _handle_event(self, event):
        if event.is_directory:
            return
            
        current_time = time.time()
        if current_time - self.last_trigger < self.cooldown:
            return
            
        self.last_trigger = current_time
        filepath = event.src_path
        print(f"\n[WATCHDOG] Detected change in: {filepath}")
        
        # Trigger ingestion and pair generation
        self._trigger_pipeline()

    def _trigger_pipeline(self):
        print("[WATCHDOG] Triggering ingestion...")
        subprocess.run(["python", "pipeline/ingest.py"])
        print("[WATCHDOG] Triggering pair generation...")
        subprocess.run(["python", "training/generate_pairs.py"])
        print("[WATCHDOG] Pipeline update complete. Ready for retraining.")

def start_watchdog(data_dir: str = "data/raw"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    event_handler = DatasetWatchdogHandler(data_dir)
    observer = Observer()
    observer.schedule(event_handler, data_dir, recursive=True)
    observer.start()
    
    print(f"[WATCHDOG] Monitoring {data_dir} for changes...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_watchdog()
