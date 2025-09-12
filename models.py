import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class SkillsetClassifier_v1_1 (nn.Module):
    def __init__(self, context_length:int, skillsets:int):
        self.SKILLSET_LABELS = ["AIM", "STREAM", "ALT", "TECH", "SPEED", "RHYTHM"]
        self.NUM_CLASSES = len(self.SKILLSET_LABELS)
        self.INPUT_DIM = 12
        self.MAX_SEQ_LEN = 124
        self.BATCH_SIZE = 16
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, (7,1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (5,1)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (5,1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (5,1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,self.INPUT_DIM))
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(256*2*self.INPUT_DIM, 256*2*self.INPUT_DIM), # (context_length - convlayerunpadding(10+10+1)) * skillsets * 32conv
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256*2*self.INPUT_DIM, 256*self.INPUT_DIM),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256*self.INPUT_DIM, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.SKILLSET_LABELS))
        )
    
    def forward(self, x, extra_x): # add extra input for metadata
        x = self.conv_block(x)
        x = self.flatten(x)
        #x = torch.concat((x,extra_x), dim=1)
        logits = self.linear_relu_stack(x)
        return logits
    

class SkillsetClassifier_v1_2 (nn.Module):
    def __init__(self, context_length:int, skillsets:int):
        self.SKILLSET_LABELS = ["AIM", "STREAM", "ALT", "TECH", "SPEED", "RHYTHM"]
        self.NUM_CLASSES = len(self.SKILLSET_LABELS)
        self.INPUT_DIM = 12
        self.MAX_SEQ_LEN = 124
        self.BATCH_SIZE = 16
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, (5,5)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3)),
            nn.ReLU()
        )

        self.flatten = nn.Flatten(start_dim=1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2*114, 2*114*2), # (context_length - convlayerunpadding(10+10+1)) * skillsets * 32conv
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*114*2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.SKILLSET_LABELS))
        )
    
    def forward(self, x, extra_x): # add extra input for metadata
        x = self.conv_block(x)
        x = self.flatten(x)
        #x = torch.concat((x,extra_x), dim=1)
        logits = self.linear_relu_stack(x)
        return logits

