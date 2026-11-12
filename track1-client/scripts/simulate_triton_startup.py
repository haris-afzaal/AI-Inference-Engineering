import os
import time

def simulate_triton_startup():
    print("=======================================")
    print("   NVIDIA Triton Inference Server      ")
    print("=======================================\n")
    
    repo_path = "artifacts/day32/model_repository"
    
    print("I1112 15:35:00.000000 1 server.cc:123] Initializing Triton Inference Server...")
    print(f"I1112 15:35:00.000000 1 model_repository_manager.cc:100] Polling model repository '{repo_path}'\n")
    
    if not os.path.exists(repo_path):
        print("[FATAL ERROR] Model repository not found!")
        return

    models = sorted(os.listdir(repo_path))
    success_count = 0
    
    for model in models:
        config_path = os.path.join(repo_path, model, "config.pbtxt")
        version_path = os.path.join(repo_path, model, "1")
        
        print(f"I1112 15:35:01.000000 1 model_repository_manager.cc:456] Loading model '{model}'...")
        time.sleep(0.5)
        
        if not os.path.exists(config_path):
            print(f"[ERROR] server.cc:300] Failed to load '{model}': Missing 'config.pbtxt'. Every model MUST have a config file.\n")
            continue
            
        if not os.path.exists(version_path):
            print(f"[ERROR] server.cc:301] Failed to load '{model}': Missing version folder (e.g., '1/').\n")
            continue
            
        print(f"I1112 15:35:02.000000 1 backend_manager.cc:789] Successfully loaded model '{model}' version 1 using 'vllm' backend.\n")
        success_count += 1
        
    print("---------------------------------------")
    if success_count > 0:
        print(f"I1112 15:35:03.000000 1 server.cc:999] Triton Server is ready! Loaded {success_count}/{len(models)} models.")
    else:
        print("[FATAL ERROR] Triton Server failed to start. No valid models loaded.")
    print("=======================================")

if __name__ == "__main__":
    simulate_triton_startup()
