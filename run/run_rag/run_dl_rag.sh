#!/bin/bash

# Define the model deployments
deployments=(
    "PATH_TO_MODEL_1"
)

# Define paths
operation_identify_folder='DS-CodeBridge/dl200_prompt_rag/operation_identify'
intermediate_output_folder='DS-CodeBridge/dl200_inter_rag'
rag_prompt_base='DS-CodeBridge/dl200_prompt_rag/rag_prompts'
final_output_folder='DS-CodeBridge/dl200_output_rag'
rag_translation_base="DS-CodeBridge/dl200_prompt_rag/rag_translation"
# Find all .jsonl files in the operation identify folder
operation_files=($(find "$operation_identify_folder" -type f -name "*.jsonl"))

# Loop through each model and each operation identify file
for model_path in "${deployments[@]}"; do
    # Extract model name from the path
    model_name=$(basename "${model_path}")
    
    # Determine GPU configuration
    if [[ "$model_name" == "Meta-Llama-3.1-70B-Instruct" || "$model_name" == "Qwen2.5-72B-Instruct" ]]; then
        gpu_config="0,1,2,3"
    else
        gpu_config="0"
    fi


    for operation_file in "${operation_files[@]}"; do
        # Extract operation filename without extension
        operation_filename=$(basename "$operation_file" .jsonl)
        direction=$(echo "$operation_filename" | cut -d'_' -f2-)
        # Step 1: Generate intermediate output
        intermediate_output_path="${intermediate_output_folder}/${model_name}_${operation_filename}.jsonl"

        echo "Generating intermediate output for ${model_name} using ${operation_filename}..."
        python3 DS-CodeBridge/src/vllm_infer.py \
            --model_path "${model_path}" \
            --prompt_path "${operation_file}" \
            --output_path "${intermediate_output_path}" \
            --gpu "${gpu_config}"

        echo "Intermediate output saved to ${intermediate_output_path}"

        # Step 2: Perform FAISS retrieval
        rag_update_prompt_path="${rag_prompt_base}/${model_name}_rag_${direction}.jsonl"
        
        echo "Performing FAISS retrieval for ${model_name} with ${operation_filename}..."
        python3 DS-CodeBridge/src/faiss_retrieval.py \
            --input_path "${intermediate_output_path}" \
            --output_path "${rag_update_prompt_path}" \
            --rag_translation_base "${rag_translation_base}"

        echo "FAISS retrieval output saved to ${rag_update_prompt_path}"

        # Step 3: Generate final output
        final_output_path="${final_output_folder}/${model_name}_${operation_filename}.jsonl"

        echo "Generating final output for ${model_name} with ${operation_filename}..."
        python3 DS-CodeBridge/src/vllm_infer.py \
            --model_path "${model_path}" \
            --prompt_path "${rag_update_prompt_path}" \
            --output_path "${final_output_path}" \
            --gpu "${gpu_config}"

        echo "Final output saved to ${final_output_path}"

        echo "----------------------------------------"
    done
done

echo "All processing completed."