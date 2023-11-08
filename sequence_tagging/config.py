class Config:
    def __init__(self):
        self.max_length = 128

        self.embedding_dim = 300
        self.hidden_dim = 512
        self.num_layers = 2
        self.dropout = 0.1

        self.epoch = 10
        self.learning_rate = 1e-3
        self.batch_size = 512

        self.save_model = "NERModel.pth"
