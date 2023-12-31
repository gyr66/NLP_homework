import os
import subprocess

perl_path = os.path.join(os.path.curdir, "semeval2010_task8_scorer-v1.2.pl")
output_predict_file = os.path.join(os.path.curdir, "predict.txt")
target_path = os.path.join(os.path.curdir, "target.txt")
process = subprocess.Popen(
    ["perl", perl_path, output_predict_file, target_path], stdout=subprocess.PIPE
)
str_parse = str(process.communicate()[0]).split("\\n")[-2]
idx = str_parse.find("%")
f1_score = float(str_parse[idx - 5 : idx])
print(f1_score)