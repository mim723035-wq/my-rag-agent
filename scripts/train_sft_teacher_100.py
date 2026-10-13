"""Fresh-base LoRA training on the frozen 100-trajectory dataset.

Sparse vocabulary projection computes exactly the supervised next-token loss;
full sequence attention is retained. --smoke-longest checks loss equivalence
and one longest-sample update without writing a usable training checkpoint.
"""
import os
os.environ["HF_HOME"]=r"D:\search-agent-cache\huggingface"
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch.nn import functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data/sft_teacher_100_v1"

def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def batch_for(row):
 return {k:torch.tensor([row[k]],dtype=torch.long,device="cuda") for k in ["input_ids","attention_mask","labels"]}

def supervised_loss(model,batch):
 # Position i predicts label i+1. Only vocabulary projection is selected;
 # input context, attention and EOS supervision are unchanged.
 positions=torch.nonzero(batch["labels"][0,1:]!=-100,as_tuple=False).flatten()
 if not positions.numel():raise ValueError("Empty shifted targets")
 targets=batch["labels"][0,1:][positions]
 output=model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],
              use_cache=False,logits_to_keep=positions)
 return F.cross_entropy(output.logits[0].float(),targets)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument("--epochs",type=int,default=1)
 parser.add_argument("--smoke-longest",action="store_true")
 args=parser.parse_args()
 if args.epochs<1:parser.error("epochs must be positive")
 report=read(DATA/"validation_report.json");manifest=read(DATA/"manifest.json")
 if report["status"]!="PASS" or report.get("token_mask")!="PASS":raise ValueError("Dataset validation incomplete")
 if sha(DATA/"train.jsonl")!=manifest["train_sha256"] or sha(DATA/"tokenized.jsonl")!=report["tokenized_sha256"]:
  raise ValueError("Frozen dataset changed; revalidate before training")
 rows=[json.loads(s) for s in (DATA/"tokenized.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
 config=read(ROOT/"configs/model.json")
 if len(rows)!=280 or len({r["metadata"]["question_id"] for r in rows})!=100:raise ValueError("Unexpected dataset")
 for row in rows:
  if row["metadata"]["model"]!=config or len(row["input_ids"])>4096:raise ValueError("Model/length mismatch")
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 CUDA is required")
 torch.manual_seed(42);rng=random.Random(42)
 kind="sft_teacher100_smoke" if args.smoke_longest else "sft_teacher100"
 stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
 run=ROOT/"outputs"/f"{kind}_{stamp}_{uuid4().hex[:8]}";run.mkdir(parents=True)
 training={"experiment":kind,"model":config,"initialization":"fresh original base; no previous adapter loaded",
    "data_file":str(DATA/"tokenized.jsonl"),"data_sha256":sha(DATA/"tokenized.jsonl"),
    "train_sha256":sha(DATA/"train.jsonl"),"trajectories":100,"action_examples":len(rows),
    "epochs":args.epochs,"seed":42,"learning_rate":1e-4,"batch_size":1,"gradient_accumulation_steps":1,
    "weight_decay":0.,"lora_r":8,"lora_alpha":16,"lora_dropout":0.,"target_modules":["q_proj","v_proj"],
    "gradient_checkpointing":True,"precision":"BF16 base; adapter runtime dtypes recorded",
    "loss":"mean supervised next-token cross entropy; vocabulary head projected only at supervised predictor positions",
    "maximum_sequence_tokens":max(len(r["input_ids"]) for r in rows),
    "limitation":"Compared with old41, data source/composition and training compute also change."}
 save(run/"training_config.json",training)
 print("Loading fresh base model...",flush=True)
 model=AutoModelForCausalLM.from_pretrained(config["model_id"],revision=config["revision"],
        dtype=torch.bfloat16,use_safetensors=True,attn_implementation="sdpa").to("cuda")
 model.config.use_cache=False
 model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=8,lora_alpha=16,
       lora_dropout=0.,target_modules=["q_proj","v_proj"],bias="none"))
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
 trainable=[p for p in model.parameters() if p.requires_grad]
 assert all("lora_" in n for n,p in model.named_parameters() if p.requires_grad)
 training["trainable_dtypes"]=sorted({str(p.dtype) for p in trainable});save(run/"training_config.json",training)
 optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0.,foreach=False)
 steps=0;started=time.perf_counter()
 try:
  # Confirm optimization preserves the installed model's ordinary masked loss.
  model.eval();short=min(rows,key=lambda r:len(r["input_ids"]));batch=batch_for(short)
  with torch.no_grad():
   ordinary=model(**batch,use_cache=False).loss.item()
   selected=supervised_loss(model,batch).item()
  if abs(ordinary-selected)>1e-5:raise ValueError(("Loss equivalence failed",ordinary,selected))
  save(run/"loss_equivalence.json",{"ordinary_masked_loss":ordinary,"selected_projection_loss":selected,
        "absolute_difference":abs(ordinary-selected),"tolerance":1e-5,"status":"PASS"})
  del batch
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model.train()
  order=list(range(len(rows)));rng.shuffle(order)
  if args.smoke_longest:order=[max(range(len(rows)),key=lambda i:len(rows[i]["input_ids"]))]
  total=len(order)*(1 if args.smoke_longest else args.epochs)
  with (run/"steps.jsonl").open("x",encoding="utf-8") as log, tqdm(total=total,desc="LoRA updates",unit="step") as progress:
   for epoch in range(1,(1 if args.smoke_longest else args.epochs)+1):
    if epoch>1:rng.shuffle(order)
    for index in order:
     row=rows[index];batch=batch_for(row);optimizer.zero_grad(set_to_none=True)
     loss=supervised_loss(model,batch)
     if not torch.isfinite(loss).item():raise RuntimeError("Nonfinite loss")
     value=loss.detach().item();loss.backward()
     norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True);optimizer.step();steps+=1
     log.write(json.dumps({"step":steps,"epoch":epoch,"question_id":row["metadata"]["question_id"],
        "action_type":row["metadata"]["action_type"],"sequence_tokens":len(row["input_ids"]),
        "supervised_tokens":row["metadata"]["supervised_tokens"],"loss_before_update":value,
        "gradient_norm_before_clipping":float(norm)})+"\n");log.flush()
     progress.update();progress.set_postfix(loss=f"{value:.4f}")
     del loss,batch
  result={"status":"smoke_pass" if args.smoke_longest else "completed","optimizer_steps":steps,
      "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,"seconds":time.perf_counter()-started,
      "capability_evaluated":False}
  if not args.smoke_longest:
   model.save_pretrained(run/"adapter",safe_serialization=True);result["adapter_dir"]=str(run/"adapter")
  else:result["limitation"]="One longest-sample update only; no adapter checkpoint produced."
  save(run/"result.json",result);print(json.dumps(result,indent=2),flush=True)
 except Exception as error:
  save(run/"result.json",{"status":"failed","completed_steps":steps,"error":f"{type(error).__name__}: {error}"})
  print("Failure log:",run,flush=True);raise
 print("Run directory:",run,flush=True)

if __name__=="__main__":main()
