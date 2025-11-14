"""
Simple PPO for Text-based Q&A
This is a simplified example to understand how PPO works for text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Simple Q&A Dataset
QA_PAIRS = {
    "What is the capital of France?": "Paris",
    "What is 2+2?": "4",
    "What color is the sky?": "blue",
    "Who wrote Romeo and Juliet?": "Shakespeare",
    "What is the largest planet?": "Jupiter",
}

# Vocabulary
VOCAB = ['<PAD>', '<START>', '<END>', 'What', 'is', 'the', 'capital', 'of', 'France', 
         'Paris', '2', '+', '4', 'color', 'sky', 'blue', 'Who', 'wrote', 
         'Romeo', 'and', 'Juliet', 'Shakespeare', 'largest', 'planet', 'Jupiter', '?']

word_to_idx = {word: idx for idx, word in enumerate(VOCAB)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

def tokenize(text):
    """Convert text to indices"""
    words = text.split()
    return [word_to_idx.get(word, 0) for word in words]

def detokenize(indices):
    """Convert indices back to text"""
    return ' '.join([idx_to_word.get(idx, '<UNK>') for idx in indices])


class SimpleActorCritic(nn.Module):
    """Simple neural network for text generation"""
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        # Actor: predicts next word
        self.actor = nn.Linear(hidden_size, vocab_size)
        
        # Critic: estimates how good the current generation is
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        x: input token indices [batch_size, seq_len]
        Returns: logits for next token, value estimate
        """
        embedded = self.embedding(x)  # [batch, seq_len, hidden]
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden]
        last_hidden = lstm_out[:, -1, :]  # [batch, hidden] - use last timestep
        
        logits = self.actor(last_hidden)  # [batch, vocab_size]
        value = self.critic(last_hidden)  # [batch, 1]
        
        return logits, value


class TextQAEnvironment:
    """Environment that checks if generated answer matches target"""
    def __init__(self):
        self.qa_pairs = QA_PAIRS
        self.questions = list(QA_PAIRS.keys())
        self.reset()
    
    def reset(self):
        """Start a new question"""
        self.current_question = random.choice(self.questions)
        self.target_answer = self.qa_pairs[self.current_question]
        self.question_tokens = tokenize(self.current_question)
        self.generated_tokens = [word_to_idx['<START>']]
        self.max_length = 5  # Max answer length
        return self.question_tokens
    
    def step(self, action):
        """
        action: next token to generate
        Returns: observation, reward, done
        """
        self.generated_tokens.append(action)
        
        # Check if done
        done = (action == word_to_idx['<END>'] or 
                len(self.generated_tokens) >= self.max_length)
        
        # Calculate reward
        if done:
            # Check if answer is correct
            generated_answer = detokenize(self.generated_tokens[1:-1])  # Remove <START> and <END>
            target_answer = self.target_answer
            
            if generated_answer.strip() == target_answer:
                reward = 10.0  # Correct answer!
            else:
                # Partial credit for correct words
                gen_words = set(generated_answer.split())
                target_words = set(target_answer.split())
                overlap = len(gen_words & target_words)
                reward = overlap * 2.0 - 2.0  # Some partial credit
        else:
            reward = -0.1  # Small penalty for each step (encourage brevity)
        
        return self.question_tokens, reward, done


def collect_rollout(env, model, device, num_steps=10):
    """
    Collect experience by generating answers
    Returns: states, actions, rewards, values, log_probs, dones
    """
    states_list = []
    actions_list = []
    rewards_list = []
    values_list = []
    log_probs_list = []
    dones_list = []
    
    # Generate multiple episodes
    for episode in range(num_steps):
        question_tokens = env.reset()
        done = False
        
        while not done:
            # Current state: question + generated so far
            state = question_tokens + env.generated_tokens
            state_tensor = torch.tensor([state], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits, value = model(state_tensor)
                logits = logits.squeeze(0)
                value = value.squeeze(0)
            
            # Sample action from policy
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Take action
            _, reward, done = env.step(action.item())
            
            # Store experience
            states_list.append(state)
            actions_list.append(action.item())
            rewards_list.append(reward)
            values_list.append(value.item())
            log_probs_list.append(log_prob.item())
            dones_list.append(done)
    
    return states_list, actions_list, rewards_list, values_list, log_probs_list, dones_list


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation
    """
    advantages = []
    gae = 0
    
    # Add terminal value
    values = values + [0]
    
    # Compute advantages backward
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        # TD error
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE
        gae = delta + gamma * lam * gae * (1 - dones[t])
        advantages.insert(0, gae)
    
    # Returns for value function
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    
    return advantages, returns


def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, returns, 
               device, epochs=3, clip_range=0.2):
    
    advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(states)):
            state = torch.tensor([states[i]], dtype=torch.long).to(device)
            
            logits, value = model(state)
            logits = logits.squeeze(0)
            value = value.squeeze(0)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(actions[i])

            ratio = torch.exp(log_prob - old_log_probs[i])

            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages[i]
            policy_loss = -torch.min(surr1, surr2)

            value_loss = F.mse_loss(value, returns[i])
            entropy = dist.entropy()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(states)
        epoch_losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")

    return epoch_losses


def train_text_qa_ppo(num_iterations=100, device='cpu'):
    print("=" * 60)
    print("Training Simple Text Q&A with PPO")
    print("=" * 60)
    
    model = SimpleActorCritic(vocab_size=len(VOCAB), hidden_size=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    env = TextQAEnvironment()
    
    all_losses = []  # <-- ADDED

    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Collect rollout
        print("Collecting experience...")
        states, actions, rewards, values, log_probs, dones = collect_rollout(
            env, model, device, num_steps=5
        )
        
        print(f"\nCollected {len(states)} experiences")
        print(f"Total reward: {sum(rewards):.2f}")
        
        advantages, returns = compute_gae(rewards, values, dones)
        
        # PPO update (but now it RETURNS epoch losses)
        print("\nUpdating policy...")
        epoch_losses = ppo_update(
            model, optimizer, states, actions, log_probs, advantages,
            returns, device, epochs=3, clip_range=0.2
        )
        
        all_losses.extend(epoch_losses)   # <-- ADDED
        
        # ------------------------------------------------
        #               EVALUATION (KEEP THIS!)
        # ------------------------------------------------
        if iteration % 10 == 0:
            print("\n" + "="*60)
            print("EVALUATION:")
            print("="*60)
            model.eval()
            with torch.no_grad():
                for question, answer in list(QA_PAIRS.items())[:3]:
                    env.current_question = question
                    env.target_answer = answer
                    question_tokens = tokenize(question)
                    generated = [word_to_idx['<START>']]
                    
                    for _ in range(5):
                        state = question_tokens + generated
                        state_tensor = torch.tensor([state], dtype=torch.long).to(device)
                        logits, _ = model(state_tensor)
                        probs = F.softmax(logits.squeeze(0), dim=-1)
                        action = torch.argmax(probs).item()
                        generated.append(action)
                        if action == word_to_idx['<END>']:
                            break
                    
                    generated_text = detokenize(generated[1:-1])
                    print(f"Q: {question}")
                    print(f"Target: {answer}")
                    print(f"Generated: {generated_text}")
                    print(f"Correct: {'✓' if generated_text.strip() == answer else '✗'}")
                    print()
            model.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # ----------------------------
    # PLOT LOSS (ADDED)
    # ----------------------------
    plt.figure(figsize=(8,5))
    plt.plot(all_losses, label="PPO Training Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("PPO Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    return model
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    model = train_text_qa_ppo(num_iterations=100, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'text_qa_ppo_model.pt')
    print("\nModel saved to 'text_qa_ppo_model.pt'")