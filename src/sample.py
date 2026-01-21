import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT(total_token=len(tokenizer), max_len=512, d=768, n_layers=12).to(device)
model.load_state_dict(torch.load("model_step_100000.pt", map_location=device))
model.eval()

def text_completion_inference(prompt, max_new_tokens, temperature, top_p, repetition_penalty):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    for _ in range(int(max_new_tokens)):
        with torch.no_grad():
            curr_len = input_ids.shape[1]
            model_input = input_ids[:, -512:] if curr_len > 512 else input_ids
            
            logits = model(model_input)
            next_token_logits = logits[:, -1, :] / temperature
            
            prev_tokens = set(input_ids[0].tolist())
            for token_id in prev_tokens:
                if next_token_logits[0, token_id] > 0:
                    next_token_logits[0, token_id] /= repetition_penalty
                else:
                    next_token_logits[0, token_id] *= repetition_penalty
            
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if tokenizer.decode(next_token[0]) == "__eou__":
                break
                
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    text_completion_inference(model, tokenizer)
