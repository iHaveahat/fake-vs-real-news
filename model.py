import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from transformers import AutoTokenizer
import pandas as pd

class WELFake(Dataset):
    def __init__(self, tokenizer, split='train', train_ratio=0.8):
        self.data = pd.read_csv('welfake.csv')
        self.data = self.data.iloc[:, 2:]
        self.tokenizer = tokenizer # AutoTokenizer.from_pretrained('bert-base-cased')
        
        # Split the data
        train_size = int(len(self.data) * train_ratio)
        if split == 'train':
            self.data = self.data.iloc[:train_size]
        elif split == 'test':
            self.data = self.data.iloc[train_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = str(self.data.iloc[idx, 0]) 
        if pd.isna(text) or text == 'nan':
            text = ""  
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True, 
            max_length=256,
            return_tensors='pt'
        )
        label = self.data.iloc[idx, 1] 
        
        sample = {
            'data': tokenized['input_ids'].squeeze(),
            # 'attention_mask': tokenized['attention_mask'].squeeze(), 
            'target': torch.tensor(label, dtype=torch.long)
        }
        return sample

        # if self.transform:
        #     sample = self.transform(sample)

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
training_data = WELFake(tokenizer)
test_data = WELFake(tokenizer, split='test')
# training_data = datasets.MNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# test_data = datasets.MNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size=28996):  # BERT's vocab size
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128) 
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(256 * 128, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
    
    def forward(self, x):
        x = self.embedding(x)  
        x = self.flatten(x)  
        logits = self.network(x)
        return logits
    
model = NeuralNetwork()
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
learning_rate = 1e-3
batch_size = 64
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, epochs, loss_fn, optimizer):
    model.train()
    for _ in range(epochs):
        count = 0
        for batch in dataloader:
            data = batch['data']
            target = batch['target']
            prediction = model(data)
            loss = loss_fn(prediction, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
            if count == 100:
                count = 0
                print("Loss is" + str(loss.item()))
        
def test_loop(dataloader, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    batches = len(dataloader)
    correct = 0
    avg_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            data = batch['data']
            target = batch['target']
            prediction = model(data)
            loss = loss_fn(prediction, target)
            avg_loss += loss
            correct += (prediction.argmax(1) == target).type(torch.float).sum().item()
    print("avg loss: " + str(avg_loss.item()/batches))
    print("percent correct: " + str(correct/size))

train_loop(train_dataloader, epochs, loss_fn, optimizer)
test_loop(test_dataloader, loss_fn)
            

