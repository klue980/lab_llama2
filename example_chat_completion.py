# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        # ctrl + k + c / ctrl + k + u
        # for input 512
                [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.
4. Montmartre: Venture into the artistic heart of Paris with a visit to Montmartre. This bohemian neighborhood, perched on a hill, has been a haven for artists and writers for centuries. Explore its charming streets, discover hidden cafes, and soak in the vibrant atmosphere that inspired the likes of Picasso and Van Gogh.
5. Sainte-Chapelle: A hidden gem on the Île de la Cité, Sainte-Chapelle is renowned for its stunning stained glass windows. Step into this Gothic masterpiece and marvel at the kaleidoscope of colors that illuminate the interior, depicting biblical stories with unparalleled brilliance.
These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.
6. Musée Rodin: Immerse yourself in the world of sculptor Auguste Rodin at the Musée Rodin. Set amidst beautiful gardens, this museum showcases Rodin's iconic works, including "The Thinker" and "The Kiss." The ambiance adds a serene touch to the art-viewing experience.
7. Seine Riverbanks: Take a leisurely stroll along the Seine Riverbanks, especially in the early evening when the city lights begin to sparkle. Witness street artists, enjoy the charming bookstalls, and savor the romantic atmosphere that defines Parisian evenings.
8. Latin Quarter: Dive into the intellectual and bohemian ambiance of the Latin Quarter. Home to the Sorbonne University, this lively district boasts narrow streets filled with bookshops, cafes, and jazz clubs. It's a perfect place to immerse yourself in the city's intellectual and artistic spirit.
9. Luxembourg Gardens: Find tranquility in the Luxembourg Gardens, a lush oasis in the heart of Paris. The meticulously manicured lawns, vibrant flowerbeds, and serene fountains create a peaceful retreat. Take a leisurely stroll or relax by the pond with a classic Parisian pastime – sailing miniature boats.
10. Musée de l'Orangerie: Art enthusiasts can revel in the beauty of the Musée de l'Orangerie, located in the Tuileries Gardens. This museum houses Claude Monet's Water Lilies series, providing an immersive experience surrounded by these iconic masterpieces.
11. Marais District: Explore the historic Marais district, known for its narrow cobblestone streets and well-preserved medieval architecture. Discover trendy boutiques, art galleries, and chic cafes nestled within the charming ambiance of this fashionable neighborhood.
12. Catacombs of Paris: For a unique and somewhat macabre experience, descend into the Catacombs of Paris. This underground ossuary holds the remains of over six million people, creating a haunting yet fascinating journey through the city's subterranean history.
13. Père Lachaise Cemetery: Pay homage to some of the world's most celebrated figures at Père Lachaise Cemetery. The final resting place of luminaries such as Oscar Wilde, Jim Morrison, and Edith Piaf, the cemetery's ornate tombs and winding paths make it a contemplative and artistic destination.
14. Musée Carnavalet: Immerse yourself in the history of Paris at the Musée Carnavalet
""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)