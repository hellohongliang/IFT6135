from models import GRU, RNN
import torch
import collections
import os

"""
Notation of this program: 
    (1) Function: predict a sequence  by using the function generate(); 
    (2) initialize variables: PATH, emb_size, hidden_size, seq_len, batch_size, 
        vocab_size, num_layers, dp_keep_prob, model_type, train_path
    (3) train_path represent the path of data file "data\\ptb.train.txt"
    (4) PATH is the path of data file of model_state_dict "GRU_SGD_LR_SCHEDULE_0\\best_params.pt"
"""
emb_size = 200
hidden_size = 1500
seq_len = 35  # 70
batch_size = 20
vocab_size = 10000
num_layers = 2
dp_keep_prob = 0.35

generated_seq_len = 35
model_type = 'RNN'  # 'RNN'


# Load model
def _load_model(model_type):
    emb_size = 200
    hidden_size = 1500
    seq_len = 35  # 70
    batch_size = 20
    vocab_size = 10000
    num_layers = 2
    dp_keep_prob = 0.35

    # Load model (Change to RNN if you want RNN to predict)
    if model_type=='RNN':
        model = RNN(emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob)
        PATH = os.path.join("RNN_ADAM_0", "best_params.pt")
    else:
        model = GRU(emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob)
        PATH = os.path.join("GRU_SGD_LR_SCHEDULE_0", "best_params.pt")

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(PATH)).cuda()
        model.eval()
    else:
        model.load_state_dict(torch.load(PATH, map_location='cpu'))
        model.eval()
    return model

def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())
    return word_to_id, id_to_word

#assign the variables with values, data is in the file of data//ptb.train.txt.
def _prepare():
    if torch.cuda.is_available():
        hidden = torch.Tensor(num_layers, batch_size, hidden_size).cuda()
        hidden = torch.nn.init.zeros_(hidden)
    else:
        hidden = torch.Tensor(num_layers, batch_size, hidden_size)
        hidden = torch.nn.init.zeros_(hidden)

    prefix = "ptb"
    data_path = "data"
    train_path = os.path.join(data_path, prefix + ".train.txt")
    word_to_id, id_2_word = _build_vocab(train_path)
    word_id = torch.LongTensor(1, batch_size).random_(0, vocab_size)  # Select the first word randomly
    return id_2_word, word_to_id, hidden, word_id

#call function to load model and prepare the variables
model = _load_model(model_type)
id_2_word, word_to_id, hidden, word_id=_prepare()

#call generate Function
samples = model.generate(word_id, hidden, generated_seq_len)

#show the result
for i in range(batch_size):
    print('The first word which is picked randomly is', end=' ')
    print(id_2_word[word_id[0][i].item()])
    print('This is the {}th predicted sentence in a mini batch:'.format(i))
    for j in range(generated_seq_len):
        print(id_2_word[samples[j][i].item()], end=' ')
    print('\n')
