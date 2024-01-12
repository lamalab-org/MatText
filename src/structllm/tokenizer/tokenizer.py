from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

train_files : [str] = ["/work/so87pot/material_db/scaleup/slice_2m.csv"]
tokenizer_save_path : str = "tokenizer-slice_2m.json"


trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(train_files, trainer)

tokenizer.save(tokenizer_save_path)
