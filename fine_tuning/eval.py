import torch
from train import WhisperModelModule,Config
import whisper
from Dataset import Dataset,WhisperDataCollatorWhithPadding
from tqdm import tqdm
import evaluate
from pathlib import Path
import pandas

checkpoint_path = "exp/checkpoint/checkpoint-epoch=0009.ckpt"
test_path = "../data/aishell/test.csv"
result_path = "exp/result/result.csv"
language="zh"
SAMPLE_RATE = 16000
cfg = Config()

Path(result_path.rstrip(result_path.split("/")[-1])).mkdir(exist_ok=True)

state_dict = torch.load(checkpoint_path)
print(state_dict.keys())
state_dict = state_dict['state_dict']
whisper_model = WhisperModelModule(cfg)
# 載入微調後的model
whisper_model.load_state_dict(state_dict)
woptions = whisper.DecodingOptions(language=language, without_timestamps=True)
wtokenizer = whisper.tokenizer.get_tokenizer(True, language=language, task=woptions.task)
dataset = Dataset(test_path, wtokenizer, SAMPLE_RATE)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding())
refs = []
res = []
for b in tqdm(loader):
    input_ids = b["input_ids"].half().cuda()
    labels = b["labels"].long().cuda()
    with torch.no_grad():
        # audio_features = whisper_model.model.encoder(input_ids)
        # out = whisper_model.model.decoder(enc_input_ids, audio_features)
        results = whisper_model.model.decode(input_ids, woptions)
        for r in results:
            res.append(r.text)

        for l in labels:
            l[l == -100] = wtokenizer.eot
            ref = wtokenizer.decode(l, skip_special_tokens=True)
            refs.append(ref)
# 寫入預測結果和ground_truth
df = pandas.read_csv(test_path)
df["asr"] = res
print(df)
df.to_csv(result_path,index=False)
# f = open(result_path,"w",encoding="utf-8")
# f.write("")
# for k, v in zip(refs, res):
#     f = open(result_path,"w",encoding="utf-8")
#     f.write()

# 計算cer
cer_metrics = evaluate.load("cer")
print("CER:",cer_metrics.compute(references=refs, predictions=res))