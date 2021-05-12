from utils.helpers import read_lines
from gector.gec_model import GecBERTModel

from fastapi import FastAPI
import uvicorn
import requests

app = FastAPI()


def getmodel():
  models_url = ['https://grammarly-nlp-data-public.s3.amazonaws.com/gector/xlnet_0_gector.th',
                'https://grammarly-nlp-data-public.s3.amazonaws.com/gector/roberta_1_gector.th']
  model_names = ['xlnet_0_gector.th','roberta_1_gector.th']
  for model_url, model_name in zip(models_url, model_names):
    r = requests.get(model_url, allow_redirects=True)
    open(model_name, 'wb').write(r.content)

def predict_for_file(input_file, model, batch_size = 32):
  test_data = read_lines(input_file)
  predictions = []
  cnt_corrections = 0
  batch = []
  
  for sent in test_data:
      batch.append(sent.split())
      if len(batch) == batch_size:
          preds, cnt = model.handle_batch(batch)
          predictions.extend(preds)
          cnt_corrections += cnt
          batch = []
  if batch:
      preds, cnt = model.handle_batch(batch)
      predictions.extend(preds)
      cnt_corrections += cnt

  
  output = "\n".join([" ".join(x) for x in predictions])    
  return {'Output: ': output, 
          'Produced overall corrections: ': cnt_corrections}


getmodel()
model = GecBERTModel(vocab_path= 'data/output_vocabulary',
                         model_paths= ['roberta_1_gector.th','xlnet_0_gector.th'],
                         max_len= 50, min_len=3,
                         min_error_probability= 0.24,
                         lowercase_tokens= 0,
                         model_name= 'roberta',
                         special_tokens_fix= 1,
                         log=False,
                         confidence= 0.24,
                         is_ensemble = 1)



@app.get('/')
def root():
    return {'hello': 'world'}

@app.post('/predict')
def predict(text: str):
  path = 'input.txt'
  with open(path, 'w') as f:
    f.writelines(text)
  return predict_for_file(path, model, 128)

'''ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)'''
