#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import string


# Importing names from the given file
# 

# In[15]:


names = []
with open('yob2018.txt', 'r') as file:
    for line in file:
        name = line.strip().split(',')[0].lower()
        names.append(name)

print(names[:5])


# In[3]:


class NameDataset(Dataset):
    def __init__(self, names):
        self.start_token = '#'
        self.end_token = '%'
        self.pad_token = ' '

        chars = string.ascii_lowercase
        self.names = [''.join([ch for ch in name.lower() if ch in chars]) for name in names]
        
        self.alphabet = self.start_token + chars + self.end_token + self.pad_token
        self.charindex = {char: idx for idx, char in enumerate(self.alphabet)}
        
        self.input_names = [self.start_token + name for name in self.names]
        self.output_names = [name + self.end_token for name in self.names]
        self.max_length = max(len(name) for name in self.input_names) + 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        input_seq = [self.charindex[char] for char in self.input_names[idx]]
        output_seq = [self.charindex[char] for char in self.output_names[idx]]

        input_tensor = F.pad(torch.tensor(input_seq, dtype=torch.long),
                             (0, self.max_length - len(input_seq)),
                             'constant', self.charindex[self.pad_token])
        
        output_tensor = F.pad(torch.tensor(output_seq, dtype=torch.long),
                              (0, self.max_length - len(output_seq)),
                              'constant', self.charindex[self.pad_token])
        
        return input_tensor, output_tensor


# In[4]:


dataset = NameDataset(names)
trainloader = DataLoader(dataset, batch_size=64, shuffle=True)


# In[5]:


class NameGenerator(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers, output_size):
        super(NameGenerator, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size 
        
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True,dropout=0.4)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        
    def forward(self, x, hidden):
        x = self.emb(x)
        x, hidden = self.lstm(x, hidden)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                  weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        return hidden


# In[6]:


model = NameGenerator(len(dataset.alphabet), 128, 256, 2, len(dataset.alphabet))
criterion = nn.CrossEntropyLoss(ignore_index=dataset.alphabet.index(' '))
optimizer = opt.Adam(model.parameters(), lr=0.001)


# In[7]:


def train_model(model, trainloader, criterion, optimizer, n_epochs=20, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        trainloader = tqdm(trainloader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch")
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0), device)
            optimizer.zero_grad()

            outputs, hidden = model(inputs, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())

            loss = criterion(outputs.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            trainloader.set_postfix(loss=running_loss / len(trainloader))

        epoch_loss = running_loss / len(trainloader)
        print(f'Epoch [{epoch+1}/{n_epochs}], Training Loss: {epoch_loss:.4f}')

    print(f'Final Training Loss: {epoch_loss:.4f}')


# In[12]:


def generate_name(model, max_length=20, temperature=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    hidden = model.init_hidden(1, device) 
    
    start_token_idx = dataset.charindex[dataset.start_token]
    input_char = torch.tensor([[start_token_idx]], dtype=torch.long).to(device)
    
    output_name = ""
    
    with torch.no_grad():
        for i in range(max_length):
            output, hidden = model(input_char, hidden)
            output = output.squeeze(0).squeeze(0)
            probabilities = F.softmax(output / temperature, dim=0).cpu().numpy()
            char_index = np.random.choice(np.arange(len(probabilities)), p=probabilities)

            if char_index == start_token_idx and i == 0:
                continue
            if char_index == dataset.charindex[dataset.end_token]:
                break

            output_name += dataset.alphabet[char_index]
            input_char = torch.tensor([[char_index]], dtype=torch.long).to(device)

    return output_name 


# In[9]:


train_model(model, trainloader, criterion, optimizer)


# In[13]:


for _ in range(10):
    print(generate_name(model, 7))


# In[ ]:




