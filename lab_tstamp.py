import time
import pandas as pd
from summarization import summarization_function
from generation import generation_function

# List to store execution times
execution_times = []

# Example usage
input_text = "Your input text goes here."

# Measure and store summarization execution time
start_time_summarization = time.time()
summarized_output = summarization_function(input_text)
end_time_summarization = time.time()
execution_time_summarization = end_time_summarization - start_time_summarization
execution_times.append(("Summarization", execution_time_summarization))

# Measure and store generation execution time
start_time_generation = time.time()
generated_output = generation_function(input_text)
end_time_generation = time.time()
execution_time_generation = end_time_generation - start_time_generation
execution_times.append(("Generation", execution_time_generation))

# Print execution times
for task, time_taken in execution_times:
    print(f"{task} Execution Time: {time_taken:.4f} seconds")

# Export to a table using pandas
df = pd.DataFrame(execution_times, columns=["Task", "Execution Time (seconds)"])
df.to_csv("execution_times_separate_files.csv", index=False)