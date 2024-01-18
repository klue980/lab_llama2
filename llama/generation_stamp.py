import time

class Llama:
    # ... (이전 코드는 그대로 유지)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            # ... (이전 인자들은 그대로 유지)
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        # ... (이후 코드는 그대로 유지)

        """
        # ... (이전 코드는 그대로 유지)

        # Measure Summarization Stage execution time
        start_time_summarization = time.time()
        logits = self.model.forward(tokens, prev_pos)
        token_logprobs = -F.cross_entropy(
            input=logits.transpose(1, 2),
            target=tokens,
            reduction="none",
            ignore_index=pad_id,
        )
        end_time_summarization = time.time()
        execution_time_summarization = end_time_summarization - start_time_summarization

        # Measure Generation Stage execution time
        start_time_generation = time.time()
        for cur_pos in range(min_prompt_len, total_len):
            # ... (이전 코드는 그대로 유지)
        end_time_generation = time.time()
        execution_time_generation = end_time_generation - start_time_generation

        if logprobs:
            token_logprobs = token_logprobs.tolist()

        # ... (이전 코드는 그대로 유지)

        return (out_tokens, out_logprobs if logprobs else None)

    # ... (이후 코드는 그대로 유지)
