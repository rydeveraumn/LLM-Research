# third party
import torch
from tiktoken.core import Encoding

# first party
from models.gpt2 import GPT2Model


def generate_text_simple(
    model: GPT2Model, idx: torch.Tensor, max_new_tokens: int, context_size: int
) -> torch.Tensor:
    # idx is a (batch, num_tokens) array of indices in this context

    for _ in range(max_new_tokens):
        # Crops the current context if it exceeds the supported context size
        # if LLM supports only 5 tokens, and the context size in 10, then
        # only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # Focuses on only the last time step, so that
        # (batch, num_tokens, vocab_size) becomes
        # (batch, vocab_size)
        logits = logits[:, -1, :]

        # (batch, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        # In a greedy way get the token with the largest
        # probability
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)

        # Append the computed token and add it to the
        # input to keep generating context
        idx = torch.cat([idx, idx_next], dim=1)

    return idx


def text_to_token_ids(text: str, tokenizer: Encoding) -> torch.Tensor:
    """
    Function to convert text to encoded token ids for
    LLMs
    """
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Encoding) -> str:
    """
    Function to convert token ids back to text for
    LLMs
    """
    flatten_token_ids = token_ids.squeeze(0)
    return tokenizer.decode(flatten_token_ids.tolist())
