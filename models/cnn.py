import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
from tqdm import trange

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(67)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using:{device}")


# This is how the input has to be in this orderi!!

#   input = (Batch Size, Number of Muscles, Width of time)

# Last dimension is always length, and is the one that'll get pooled, keep in mind if we change data input

# NOTE:
# I decided to not use padding, for sligth performance boost and stricter evaluation

class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=6 #number of muscle semg inputs,
                               ,out_channels=24, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=24, out_channels=24, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(24)
        # F.maxPool1d will be applied, so take that in count 
        self.conv3 = nn.Conv1d(in_channels=24, out_channels=48, kernel_size=9)
        self.bn2 = nn.BatchNorm1d(48)
        self.conv4 = nn.Conv1d(in_channels=48, out_channels=72, kernel_size=16)
        self.bn3 = nn.BatchNorm1d(72)
        # Another maxPool1d
        # I am assuming we start with a 200ms time step

        # If you want to make it consume less RAM!!!
        # Most of the parametres come from fc1
        #So add:
        #self.pool1 = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear( 37 * 72, 256)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2) # This is reducing the time length by half

        # Second block
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2) # This is reducing the time length by half to 50

        # MLP Part
        x = torch.flatten(x, 1)
        #For the RAM reduction up there
        # x = pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

model = cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),weight_decay=0.0001 ,lr=0.001) #Lower learning 10^2 if using deistillation

# If we are going to use distillation knowledge, then use this
'''
def distillation_loss(student_logits, teacher_logits, true_labels, T=4.0, alpha=0.5):

    # Hard target loss (normal supervised)
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # Soft target loss (teacher guidance)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean"
    ) * (T * T)

    return alpha * hard_loss + (1 - alpha) * soft_loss
'''


'''
Yeah guys, I know rn the whole loading the data is missing,
I know, just put the data in the bag bro,
we'll get there
'''

for epoch in trange(5):
    # tqdm is just the progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(images)
        loss = criterion(y, labels)

        #If doing the distillation thing, of course remove the forward pass above
        '''
        student_logits = model(images)
        with torch.no_grad():
            teacher_logits = teacher(images)

        loss = distillation_loss(student_logits, teacher_logits, labels,
                                 T=4.0, alpha=0.5)
        '''
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress bar with loss info
        progress_bar.set_description(f"Training (loss={loss.item():.4f})")
