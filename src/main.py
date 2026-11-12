# Run the complete StyleSense AI text-mining pipeline

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name, task_name):
    print(f"\n======= Running {task_name} =======", flush=True)
    script_path = os.path.join(SRC_DIR, script_name)
    subprocess.run([sys.executable, script_path], check=True)

def main():
    print("\n======= Starting StyleSense AI Pipeline =======", flush=True)
    run_script("numerical.py", "Preprocessing and Feature Extraction")
    run_script("classification.py", "Classification")
    run_script("clustering.py", "Clustering")
    run_script("topic_modeling.py", "Topic Modeling")
    print("\n======= All tasks completed successfully =======", flush=True)

if __name__ == "__main__":
    main()
