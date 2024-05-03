class SlidingWindowMultiHeadAttention(nn.module):

    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisble by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size/num_heads
        self.window_size = window_size
        self.qkv_linear = nn.Linear(hidden_size, hidden_size*3)