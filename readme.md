## Description

This is the preview codebase for the benchmark DS-CodeBridge. Contains the prompt folder, example output, and the source code.

## Environment Setup
```bash
conda create -n ds-codebridge python=3.10
conda activate ds-codebridge
pip install -r requirements.txt
```

### Postgresql Setup
Download and install the postgresql from the official website: https://www.postgresql.org/download/
Download the pgAdmin4 from the official website: https://www.pgadmin.org/download/ (Recommended to monitor the database)
In pgADmin4/terminal create a new database you prefer
Construct the database by run the following command (You can find PostgreSQL & Pandas version database in the HF dataset):
```bash
psql -U USERNAME -d DB_NAME -f postgresql_db.sql
```

## Generate the prompt
Use the `prompt_generator.py` under each category folder of prompt folder to generate the prompt.
```bash
python prompt_generator.py --data_path <data_path> --prompt_base <prompt_base>
```

## Inference
Use the script [`gpt_infer.py`](./src/gpt_infer.py) to inference with proprietary model such as GPT-4 and Claude-3.5-Sonnet. You have to set your API key in the script. Use the script [`vllm_infer.py`](./src/vllm_infer.py) to inference with local models with VLLM API. Make sure to set the correct `prompt_folder` and `output_folder` in the `run_inference.sh` script.
```bash
cd run/run_baseline_intent
bash run_inference.sh
```

For RAG version, use scripts under the 'run/run_rag' folder. Make sure also set the correct path in each `run_xx_rag.sh` script. Also please download the embedding model stella_en_400M_v5 from https://huggingface.co/dunzhang/stella_en_400M_v5


## Evaluation
Use the script [`evaluation_EA.py`](./eval/dl200/evaluation_EA.py) under each category to evaluate the correspond generated code. You have to set the correct `ground_truth_path` and `predictions_path` in the argument.
```bash
python evaluation_EA.py --ground_truth_path <ground_truth_path> --predictions_path <predictions_path>
```
