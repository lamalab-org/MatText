from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

train_files : [str] = ["/home/so87pot/n0w0f/structllm/data/130k/slice.csv"]
tokenizer_save_path : str = "tokenizer-slice_130k.json"


trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(train_files, trainer)

tokenizer.save(tokenizer_save_path)
