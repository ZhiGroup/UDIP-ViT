from models.model_avg_pool import AVGPOOLMODEL
import torch.optim as optim # for AdamW loss
from torch.optim.lr_scheduler import StepLR

class TrainValObjAvgPool():
    def __init__(self, lr, gamma, step_size, embed_dim, decoder_embed_dim, num_heads):
        self.lr = lr
        self.gamma = gamma
        self.step_size = step_size
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.num_heads = num_heads

    def load_train_val_obj(self):
        model = AVGPOOLMODEL(embed_dim = self.embed_dim, decoder_embed_dim = self.decoder_embed_dim, num_heads = self.num_heads)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return model, optimizer, scheduler
