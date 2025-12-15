```mermaid
graph TD
    %% Nodes
    A[("NewsScape Archive<br>(.mp4 / .mkv)")] -->|Raw Video| B["Audio Extraction"]
    B -->|"Audio Stream"| C{"Whisper Model<br>(ASR)"}
    
    subgraph HPC_Cluster ["CWRU HPC Cluster"]
        style HPC_Cluster fill:#f9f,stroke:#333,stroke-width:2px
        C -->|"Transcribed Text"| D["Singularity Container"]
        D -->|"Tokenized Input"| E["Llama-3-8B<br>(4-bit Quantized)"]
        E -->|"Inference"| F["Frame Blending Analysis"]
    end

    F -->|"Structured Data"| G[("Output JSON<br>(Metadata)")]

    %% Styling
    classDef storage fill:#eee,stroke:#333,stroke-width:2px;
    classDef process fill:#d4e1f5,stroke:#333,stroke-width:2px;
    class A,G storage;
    class B,C,D,E,F process;
```
## 🛠️ Tech Stack
* **Containerization:** Apptainer / Singularity
* **Model:** Llama-3-8B (4-bit Quantized via `bitsandbytes`)
* **Orchestration:** Slurm Workload Manager
* **Language:** Python 3.9+
* **Optimization:** `bitsandbytes` (NF4 Quantization) to reduce VRAM usage by ~60%.  