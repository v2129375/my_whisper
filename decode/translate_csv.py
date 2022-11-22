import pandas
import whisper
from tqdm import tqdm

class ASR_CSV:
    def __init__(self):
        self.model = whisper.load_model("large")
    def asr_csv(self,input_path,output_path):
        df = pandas.read_csv(input_path)
        df["asr"] = ""
        for i in tqdm(range(len(df))):
            asr_text = self.model.transcribe(df["path"][i])["text"]
            df.loc[i,"asr"] = asr_text
            print(asr_text)
        df.to_csv(output_path,index=False)

if __name__ == '__main__':
    a = ASR_CSV()
    a.asr_csv("/Data/AmbulancePaper/myCATSLU/data/all/train.csv","train.csv")
    a.asr_csv("/Data/AmbulancePaper/myCATSLU/data/all/dev.csv", "dev.csv")
    a.asr_csv("/Data/AmbulancePaper/myCATSLU/data/all/test.csv", "test.csv")

