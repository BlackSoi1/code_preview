#!/bin/bash

# Define the model deployments
deployments=(
    "PATH_TO_MODEL_1"
)

# Define the prompt folder and output folder
prompt_folder='DS-CodeBridge/XX_prompt/XX_intent{baseline}'
output_folder='DS-CodeBridge/XX_output/XX_intent{baseline}'

# Find all .jsonl prompt files in the prompt folder
prompt_paths=($(find "$prompt_folder" -type f -name "*.jsonl"))

# Loop through each model and each prompt file
for model_path in "${deployments[@]}"; do
    # Extract model name from the path
    model_name=$(basename "${model_path}")
    
    for prompt_path in "${prompt_paths[@]}"; do
        # Extract prompt filename without extension
        prompt_filename=$(basename "$prompt_path" .jsonl)
        
        # Construct output path
        output_path="${output_folder}/${model_name}_${prompt_filename}.jsonl"

        # Check if output file already exists, skip if it does
        if [ -f "$output_path" ]; then
            echo "Output file ${output_path} already exists. Skipping..."
            continue
        fi
        
        echo "Processing model: ${model_name}"
        echo "Model path: ${model_path}"
        echo "Using prompt: ${prompt_filename}"
        echo "Output path: ${output_path}"
        
        # Run the Python script
        python3 DS-CodeBridge/src/vllm_infer.py --model_path "${model_path}" --prompt_path "${prompt_path}" --output_path "${output_path}" --gpu "0,1,2,3"
        
        echo "Completed processing for ${model_name} with ${prompt_filename}"
        echo "----------------------------------------"
    done
done

echo "All processing completed."