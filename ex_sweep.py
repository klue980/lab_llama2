
# title : llama2/ex_sweep.py
# description : iteration script with example_chat_completion.py of llama2
# author : Kim Seong Ho
# email : klue980@gmail.com 
# since : 2024.01.19
# update  : 2024.01.19

# ex_sweep : iteration script with example_chat_completion.py of llama2
# use subprocess.run to repeatedly put CLI commands

import subprocess

# sweep parameter
max_seq_len_values = []
max_batch_size_values = []

for i in range(5): # for 0, 1, 2, 3, 4
    max_seq_len_values.append(1024 + 32*(2**i))
    max_batch_size_values.append(1 * (4**i))

# base command
base_command = "torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model"

# iteration script
for max_seq_len in max_seq_len_values:
    for max_batch_size in max_batch_size_values:
        command = f"{base_command} --max_seq_len {max_seq_len} --max_batch_size {max_batch_size}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"error : {e}")