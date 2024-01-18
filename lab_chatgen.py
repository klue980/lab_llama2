import pandas as pd
from typing import List, Optional

from llama import Llama, Dialog
import fire

def generate_and_save_results(
    ckpt_dir: str,
    tokenizer_path: str,
    batch_sizes: List[int],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: Optional[int] = None,
):
    """
    Generate text using a pretrained model for different batch sizes and save results to a CSV file.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        batch_sizes (List[int]): List of batch sizes to sweep through.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """

    # Initialize the generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max(batch_sizes),
    )

    # Store results in a list
    results_list = []

    # Iterate over different batch sizes
    for batch_size in batch_sizes:
        dialogs: List[Dialog] = [
            # Add your example dialog here
        ]
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )

        # Store results in a dictionary
        result_dict = {
            "Batch Size": batch_size,
            "Dialog": dialogs,
            "Result": results
        }

        # Append result dictionary to the list
        results_list.append(result_dict)

    # Convert results to DataFrame
    df = pd.DataFrame(results_list)

    # Save results to a CSV file
    df.to_csv("generation_results.csv", index=False)

if __name__ == "__main__":
    # Example usage:
    # python script_name.py --ckpt_dir your_checkpoint_directory --tokenizer_path your_tokenizer_path --batch_sizes 4 8 16
    fire.Fire(generate_and_save_results)