import torch
import torch.nn as nn
import pandas as pd
import math
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import nltk
import re
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import torch.optim as optim

#nltk.download('punkt') 




#Standard embedding 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(InputEmbeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    #(batch,seq_len) -> (batch,seq_len,d_model)
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# Positional encoding block with the formulae used in the original paper

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)  # requires_grad_(False) as we are not learning this tensor
        return self.dropout(x)

#Standard layer normalization implementation

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

#This block corresponds to the feed forward block of the encoder that comes after the multi head attention block

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

#This block has the encoder output as input and transforms it into a vector of length 2 (as there are two classes in our problem) whose bigger coordinate corresponds to the predicted class

class EndNetwork(nn.Module):

    def __init__(self, d_model, seq_len) -> None:
        super().__init__()
        self.linear = nn.Linear(seq_len*d_model,2)
        self.seq_len = seq_len
        self.d_model = d_model

    # (batch, seq_len, d_model) -> (batch, seq_len*d_model, 1) -> (batch, 2)
    def forward(self,x):
        return self.linear(x.reshape(-1,self.seq_len*self.d_model))



class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) 
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return attention_scores @ value

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


#Single Encoder Block (6 were stacked in the original paper)
class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

#We stack the encoder blocks to create the whole Encoder module; the "stacker" module is already implemented in pytorch as nn.ModuleList

class Encoder(nn.Module): 

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

#Highest module in the hierarchy. This block defines a Transformer classifier: It connects embedding, positional encoding , encoder, and final FFN to produce a classifier model.

class Modele(nn.Module):

    def __init__(self, encoder: Encoder, embed: InputEmbeddings,
                 pos: PositionalEncoding, output: EndNetwork) -> None:
        super().__init__()
        self.encoder = encoder
        self.embed = embed
        self.pos = pos
        self.output = output

  
    def forward(self, input, mask=None):
        # (batch, seq_len, d_model)
        x = self.embed(input)
        x = self.pos(x)
        x = self.encoder(x, mask)
        return self.output(x) # (batch, 2)
    
    

# Model constructor. After the definition, we initialize a model.

def build_modele(vocab_size: int, seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Modele:
    # Create the embedding layers
    embed = InputEmbeddings(d_model, vocab_size)

    # Create the positional encoding layers
    pos = PositionalEncoding(d_model, seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the encoder 
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Create the end of the network
    output = EndNetwork(d_model, seq_len)

    # Create the model
    modele = Modele(encoder, embed, pos, output)

    # Initialize the parameters
    for p in modele.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return modele

#We initialize a small model with appropiate parameters for the task. User is invited to play with the parameters. NOTE: number of head (h) has to divide d_model.
model = build_modele(vocab_size = 26364, seq_len= 39, d_model = 64, N = 1, h = 2, dropout = 0.1, d_ff = 128)

################TESTING#######################
# We set the device to CUDA if it is available for faster training

if torch.cuda.is_available():
    device = torch.device("cuda")  # Use GPU
else:
    device = torch.device("cpu")   # Use CPU

device = torch.device("cpu") 

# Move the model to the device
model.to(device)

# Set the model to evaluation mode
model.eval()


####Dataset import and processing#####

DataPath = 'C:\Master\S2\Projet DL\Sarcasm_Headlines_Dataset_v2.csv'

df = pd.read_csv(DataPath)
print(df.head(), "\n")

column_names = df.columns.tolist()
print("Column names:", column_names)

df = df.drop(columns=['article_link'])

##During the following sections, dataset is transformed sequentially: We tokenize our data, and then we encode it 
#and apply the appropiate padding for the embedding layer to work

##### Dataset tokenization:####
pattern = r'\b\w+\b' #Specific pattern that matches words in a string. We use it to tokenize our data

df_tok = df.copy() 

for i in range(len(df_tok)):
    headline = df_tok.iloc[i]['headline']
    tokenized_headline = re.findall(pattern, headline.lower())
    df_tok.at[i, 'headline'] = tokenized_headline

#We decided to investigate the length of the headlines

sorted_df = df_tok.copy()
sorted_df['headline_length'] = sorted_df['headline'].apply(len)
sorted_df = sorted_df.sort_values(by='headline_length', ascending=False)

# Get the lengths of the ten longest headlines
ten_longest_lengths = sorted_df['headline_length'].head(10)

# Print the lengths of the ten longest headlines
print("Lengths of the ten longest headlines:")
print(ten_longest_lengths)
# We drop the longest headline as it was an extreme outlier
max_length_index = df_tok['headline'].apply(len).idxmax()
df_tok = df_tok.drop(index=max_length_index)


#Here we encode our words to prepare the 
distinct_words = set()

# Iterate over each headline in the DataFrame
for headline in df_tok['headline']:
    # Flatten the list of words and add them to the set
    distinct_words.update(headline)

# Count the number of distinct words
num_distinct_words = len(distinct_words)

# Print the number of distinct words
print("Number of distinct words:", num_distinct_words)

print(df_tok.iloc[0]['headline'])

# Create a new DataFrame df_encoded as a copy of df_tok
df_enc = df_tok.copy()

# Initialize variables
word2value = {'.':0} #Text has been already tokenized so no '.' will be found. So 0 will serve as padding.
current_value = 1

# Iterate over each row in the DataFrame
for index, row in df_enc.iterrows():
    # Initialize encoded_headline list
    encoded_headline = []
    
    # Iterate over each word in the headline
    for word in row['headline']:
        
        if word not in word2value:
            # Encode the word and store it in the dictionary
            word2value[word] = current_value
            current_value += 1
        
        # Append the encoded value to the encoded_headline list
        encoded_headline.append(word2value[word])
    
    # Update the headline with encoded values
    df_enc.at[index, 'headline'] = encoded_headline

print(df_enc)
print("\nWord to value mapping:")
print(word2value)
print(type(df_enc.iloc[0]['headline']))

#In this section we want to pad every encoded vector via using the pad_sequence function from torch.nn.utils.rnn. So we save the 'headline' column as a list of tensors
#and reconstruct a dataframe afterwards

headline_tensors = [torch.tensor(encoded_headline) for encoded_headline in df_enc['headline']]


padded = pad_sequence(headline_tensors, batch_first=True, padding_value=0).narrow(1,0,39)
print(len(padded))

#With these prints we checked our data had the appropiate dimension for our model
#print(padded.shape)
#print(padded)

#In this part we 'reconstruct' the dataframe
df_enc = pd.DataFrame({'is_sarcastic': [0]*28618})  

# Initialize lists to store values for the DataFrame
headlines = []
is_sarcastic_values = []

# Iterate over each row of the tensor
for i in range(padded.size(0)):
    # Extract the row from the tensor
    row = padded[i]
    
    # Convert the tensor row to a list and append it to the 'headlines' list
    headlines.append(row.tolist())
    # Extract the corresponding value of 'is_sarcastic' column from the other DataFrame
    is_sarcastic_value = df_tok.iloc[i]['is_sarcastic']
    # Append the value to the 'is_sarcastic_values' list
    is_sarcastic_values.append(is_sarcastic_value)

# Final dataframe
final_df = pd.DataFrame({'headlines': headlines, 'is_sarcastic': is_sarcastic_values})

# We check that everything is in order
#print(final_df)

####### Train-test split ########

# Split the data into training and testing sets (80% training, 20% testing)
train_df, test_df = train_test_split(final_df, test_size=0.15, random_state=42)

print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)


### Training loop ###

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract features and labels for a single row
        features = torch.tensor(self.dataframe.iloc[idx]['headlines'], dtype=torch.int)  # Assuming 'headlines' column contains features
        label = torch.tensor(self.dataframe.iloc[idx]['is_sarcastic'], dtype=torch.long)  # Assuming 'is_sarcastic' column contains labels
        return features, label

# Define batch size and number of epochs
batch_size = 25
n_epochs = 6

# Create custom datasets
train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model, criterion, and optimizer
# Replace these with your actual model, loss function, and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)


# Standard Training loop
for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    # Mini-batch training
    print(epoch)
    i = 0
    for x, y in train_loader:
        # Send data to the correct device (you may be using CUDA)
        inputs, labels = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        i += 1
        if i % 10 == 0:
            print("Batch", i, "Epoch", epoch, "loss_batch", loss.item())
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}')

# Validation loop
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')

#The following commented code was used to produce both plots used in the report.
#It works by training the model normally util it has done one epoch, then setting the model in evaluation mode and calculating accuracy on test and train sets, 
#Then train mode is activated and the process is repeated until it has been trained on n_epochs

"""Loss_val = []
TEST = []
TRAIN = []
 
for epoch in range(n_epochs):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        TEST.append(accuracy)
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        TRAIN.append(accuracy)
    model.train()  # Set the model to training mode
    running_loss = 0.0
    # Mini-batch training
    print(epoch)
    i = 0
    for x, y in train_loader:
        # Send data to the correct device (you may be using CUDA)
        inputs, labels = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        i += 1
        if i % 10 == 0:
            print("Batch", i, "Epoch", epoch, "loss_batch", loss.item())
            Loss_val.append(loss.item())
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}')


model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    TEST.append(accuracy)
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    TRAIN.append(accuracy)     

plt.plot(TRAIN,'r')
plt.plot(TEST,'g')

plt.show()

plt.plot(Loss_val)
plt.show()"""