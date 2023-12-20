from transformers import AutoModel, AutoTokenizer
from huggingface_hub import create_repo, Repository

model = AutoModel.from_pretrained("/mnt/ds3lab-scratch/yirguo/models/checkpoint-188500")
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/ds3lab-scratch/yirguo/models/checkpoint-188500",
    src_lang="en_XX",
    tgt_lang="zh_CN",
)

model.config.decoder_start_token_id = tokenizer.lang_code_to_id["zh_CN"]


model.save_pretrained("/mnt/ds3lab-scratch/yirguo/models/epoch_9")
tokenizer.save_pretrained("/mnt/ds3lab-scratch/yirguo/models/epoch_9")

repo_name = "machine_translation"
# repo_url = create_repo(repo_name, private=True)
repo_url = "https://huggingface.co/gyr66/machine_translation"

repo = Repository(
    local_dir="/mnt/ds3lab-scratch/yirguo/models/machine_translation",
    clone_from=repo_url,
)
with repo.commit("Add model and tokenizer (Epcoh 9)"):
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
