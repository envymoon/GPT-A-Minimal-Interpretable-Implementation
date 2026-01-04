import torch
import torch.nn as nn
import json
from dataset import get_dataloaders
from model import GPT
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


if __name__ == "__main__":
    train_dataloader, val_dataloader, tokenizer = get_dataloaders(batch_size=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = GPT(total_token=len(tokenizer), max_len=512, d=768, n_layers=12).to(device)
    model.to(device)
    model.load_state_dict(torch.load("model_step_100000.pt", map_location=device))
    
    num_warmup_steps = 2000 
    total_steps = 690413 
    global_step = 0
    eval_every = 100
    save_every = 2000 
    log_every = 100 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    CEL = nn.CrossEntropyLoss()
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_steps
    )
    
    history = {
        "train_loss": [],
        "steps": [],
        "val_steps": [],
        "val_loss": []
    }    
    
    train_iter = iter(train_dataloader)
    model.train()
    
    with tqdm(range(global_step, total_steps), desc="Training") as pbar:
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            input_token = input_ids[:, :-1]
            # Shifted Alignment
            target_token = input_ids[:, 1:]
            
            optimizer.zero_grad()
            logits = model(input_token)
            loss = CEL(logits.reshape(-1, model.vocab_size), target_token.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}", step=step)
            
            if step % log_every == 0:
                history["train_loss"].append(loss.item())
                history["steps"].append(step)
                    
            if step % 1000 == 0:
                with open("../data/train_history.json", "w") as f:
                    json.dump(history, f)
            
            if step % save_every == 0 and step > 0:
                torch.save(model.state_dict(), f"model_latestet.pt") #model_step_{step}.pt
        
            if step % eval_every == 0:
                model.eval()
                total_val_loss = 0
                val_count = 0
                with torch.no_grad():
                    for i, v_batch in enumerate(val_dataloader):
                        v_input = v_batch['input_ids'].to(device)
                        v_logits = model(v_input[:, :-1])
                        v_loss = CEL(v_logits.reshape(-1, model.vocab_size), v_input[:, 1:].reshape(-1))
                        total_val_loss += v_loss.item()
                        val_count += 1
                        if i > 50: break
                    
                avg_val_loss = total_val_loss / val_count
                history["val_steps"].append(step)
                history["val_loss"].append(avg_val_loss)
                    
                with open("../data/train_history.json", "w") as f:
                    json.dump(history, f)
            
            global_step = step