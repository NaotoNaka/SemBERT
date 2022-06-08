import json
import numpy as np
import tensorflow_hub as hub
import pickle
from itertools import chain

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class readJson():
  def __init__(self,path):
    self.path=path
    decoder = json.JSONDecoder()
    self.annotation=[]
    with open(path, 'r')as f:
      line=f.readline()
      while line:
        self.annotation.append(decoder.raw_decode(line))
        line=f.readline()
    #print([i[0]["context_label"]for i in self.annotation])
  def getInfo(self,index):
    return self.annotation[index]
  def getData(self):
    return self.annotation
  def getLen(self):
    return len(self.annotation)
def getARP(TFPN):
  print(TFPN)
  TP,FP,FN,TN=[i+1e-3 for i in TFPN]

  print("Accuracy: "+str((TP+TN)/(TP+FP+TN+FN)))
  print("Recall: "+str(TP/(TP+FN)))
  print("Precision: "+str(TP/(TP+FP)))
  print("P(label=OOC|Pred=OOC): "+str(TP/(TP+FN)))
  print("P(label=NOOC|Pred=NOOC): "+str(TN/(FP+TN)))
  print("-"*100)
def baseline(bbox,bert):
  if(bbox):
    if(bert):
      return 1
    else:
      return 0
  else:
    return 0
def modified(sembert, bbox, bert):
  if(sembert):
    return 1
  else:
    if(bbox):
      if(bert):
        return 1
      else:
        return 0
    else:
      return 0
if(__name__=="__main__"):
  with open('/home/nnaka/home/cheapfake-challenge-acmmm-22/COSMOS/COSMOS.pickle', 'rb') as handle:
    outputs=pickle.load(handle)
  with open('logitLog.pickle', 'rb') as handle:
    decode=pickle.load(handle)
    logitLog = [i[0].tolist() for i in decode]
    labelLog = [i[1].tolist() for i in decode]
  logitLog=list(chain.from_iterable(logitLog))
  labelLog=list(chain.from_iterable(labelLog))
  #print(labelLog)
  embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
  path = '/net/per920a/export/das14a/satoh-lab/y-kondo/cheapfake-challenge-acmmm-22/data/cosmos_anns_acm/acm_anns/public_test_acm.json'
  jr=readJson(path)
  idx=1
  #print(jr.getInfo(0))
  #print(logitLog[0])
  #print(labelLog[0])
  #print([i[0]['context_label'] for i in jr.getData()])
  #print(outputs)
  #print(1/0)
  TFPN=[0]*24
  for data,sembert,bbox in zip(jr.getData(),logitLog,outputs):
    embeddings = embed([data[0]["caption1_modified"],data[0]["caption2_modified"]])
    sim=cos_sim(embeddings[0], embeddings[1])
    bert=float(data[0]['bert_base_score'])<0.5
    use=float(sim)<0.5
    if(data[0]['context_label']):#OOC
      if(bert):
        TFPN[0]+=1
      else:
        TFPN[1]+=1
      if(use):
        TFPN[4]+=1
      else:
        TFPN[5]+=1
      if(sembert):
        TFPN[8]+=1
      else:
        TFPN[9]+=1
      if(bbox):
        TFPN[12]+=1
      else:
        TFPN[13]+=1
      if(modified(sembert, bbox, use)):
        TFPN[16]+=1
      else:
        TFPN[17]+=1
      if(baseline(bbox,use)):
        TFPN[20]+=1
      else:
        TFPN[21]+=1
    else:
      if(bert):
        TFPN[2]+=1
      else:
        TFPN[3]+=1
      if(use):
        TFPN[6]+=1
      else:
        TFPN[7]+=1
      if(sembert):
        TFPN[10]+=1
      else:
        TFPN[11]+=1
      if(bbox):
        TFPN[14]+=1
      else:
        TFPN[15]+=1
      if(modified(sembert, bbox, use)):
        TFPN[18]+=1
      else:
        TFPN[19]+=1
      if(baseline(bbox,use)):
        TFPN[22]+=1
      else:
        TFPN[23]+=1
  print("TP,FP,FN,TN of BERT")
  getARP(TFPN[:4])
  print("TP,FP,FN,TN of USE")
  getARP(TFPN[4:8])
  print("TP,FP,FN,TN of SemBERT")
  getARP(TFPN[8:12])
  print("TP,FP,FN,TN of BBox")
  getARP(TFPN[12:16])
  print("TP,FP,FN,TN of modified")
  getARP(TFPN[16:20])
  print("TP,FP,FN,TN of baseline")
  getARP(TFPN[20:24])