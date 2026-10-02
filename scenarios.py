#!/usr/bin/env python3
"""Actual OpenAI streaming scenarios, retained request timings and exact token counts."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, importlib.util, json, math, statistics, threading, time, uuid
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
MODEL='g53fq'
spec=importlib.util.spec_from_file_location('needle','/home/josh/llm-inference-bench/needle_gate.py')
needle=importlib.util.module_from_spec(spec);spec.loader.exec_module(needle)

def session():
    s=requests.Session();s.trust_env=False;return s

def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def quantile(values,q):
    if not values:return None
    vals=sorted(values);x=(len(vals)-1)*q;low=math.floor(x);high=math.ceil(x)
    return vals[low]+(vals[high]-vals[low])*(x-low)

def stream(port,prompt,*,tokens=1024,seed=731,chat=False,timeout=300,first_event=None,ignore_eos=True):
    body={'model':MODEL,'temperature':0.6 if not chat else 0,'max_tokens':tokens,'seed':seed,'stream':True,'stream_options':{'include_usage':True}}
    if chat:
        body['messages']=[{'role':'user','content':prompt}];body['chat_template_kwargs']={'reasoning_effort':'low'}
    else:body.update(prompt=prompt,ignore_eos=ignore_eos)
    path='/v1/chat/completions' if chat else '/v1/completions'
    started=time.time();first=None;last=None;events=[];content=[];reason=[];usage=None;finish=None
    result={'port':port,'started':started,'seed':seed,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt_chars':len(prompt),'max_tokens':tokens,'chat':chat,'ignore_eos':ignore_eos if not chat else None}
    try:
        with session() as s, s.post(f'http://127.0.0.1:{port}'+path,json=body,stream=True,timeout=(10,timeout)) as response:
            result['http_status']=response.status_code
            if not response.ok:raise RuntimeError(response.text[:1500])
            for raw in response.iter_lines(chunk_size=None):
                if time.time()-started>timeout:raise TimeoutError('Total request deadline exceeded')
                if not raw.startswith(b'data: '):continue
                text=raw[6:]
                if text==b'[DONE]':break
                data=json.loads(text)
                if data.get('error'):raise RuntimeError(str(data['error']))
                if data.get('usage'):usage=data['usage']
                for choice in data.get('choices',[]):
                    delta=choice.get('delta',{})
                    visible=choice.get('text') or delta.get('content') or ''
                    thinking=delta.get('reasoning_content') or delta.get('reasoning') or ''
                    if visible or thinking:
                        now=time.time();first=first or now;last=now;events.append(now)
                        content.append(visible);reason.append(thinking)
                        if first_event:first_event.set()
                    if choice.get('finish_reason'):finish=choice['finish_reason']
        result.update(ok=bool(usage and finish),usage=usage,finish_reason=finish)
        if not result['ok']:result['error']='Missing final usage or finish reason'
    except Exception as exc:result.update(ok=False,error=repr(exc),usage=usage,finish_reason=finish)
    ended=time.time();n=(usage or {}).get('completion_tokens',0)
    result.update(ended=ended,elapsed_s=ended-started,ttft_s=first-started if first else None,tpot_ms=((last-first)/(n-1)*1000) if first and last and n>1 else None,
                  event_timestamps=events,max_inter_chunk_gap_s=max((b-a for a,b in zip(events,events[1:])),default=None),content=''.join(content),reasoning_chars=sum(map(len,reason)))
    return result

def summary(rows,started,ended):
    ok=[r for r in rows if r['ok']]
    tokens=sum((r.get('usage') or {}).get('completion_tokens',0) for r in ok)
    return {'requests':len(rows),'completed':len(ok),'failed':len(rows)-len(ok),'wall_s':ended-started,'output_tokens':tokens,'aggregate_output_tokens_per_s':tokens/(ended-started) if ended>started else None,
            'ttft_p50_s':quantile([r['ttft_s'] for r in ok if r['ttft_s'] is not None],.5),'ttft_p95_s':quantile([r['ttft_s'] for r in ok if r['ttft_s'] is not None],.95),
            'tpot_p50_ms':quantile([r['tpot_ms'] for r in ok if r['tpot_ms'] is not None],.5),'tpot_p95_ms':quantile([r['tpot_ms'] for r in ok if r['tpot_ms'] is not None],.95)}

def corpus(port,tokens,identity):
    needle.SALT=identity
    base=f'http://127.0.0.1:{port}'
    low=0;high=max(1,tokens//25);counts={}
    def count(n):
        if n not in counts:
            p=needle.build_prompt_from_paragraphs(n)
            with session() as s:
                r=s.post(base+'/tokenize',json={'model':MODEL,'messages':[{'role':'user','content':p}],'chat_template_kwargs':{'reasoning_effort':'low'}},timeout=180)
                r.raise_for_status();counts[n]=r.json()['count']
        return counts[n]
    while count(high)<tokens:low=high;high*=2
    while low+1<high:
        mid=(low+high)//2
        if count(mid)<=tokens:low=mid
        else:high=mid
    return needle.build_prompt_from_paragraphs(low),count(low)

TOPICS=['implement a persistent B-tree with transactions','design a deterministic job scheduler with cancellation','explain a lock-free ring buffer and memory ordering','implement a streaming CSV parser with quoted newlines','design a content-addressed build cache','explain a database write-ahead log and recovery','implement an HTTP parser for chunked transfers','design a replicated key-value store']

def throughput(ports,record):
    record['definition']='Fixed 1024-output-token raw completion bursts, 4 requests per concurrency slot; end-to-end aggregate includes prompt and queue latency. Same seeds/prompts across configurations. Not an accuracy score.'
    record['cells']=[]
    for repeat in range(3):
        for conc in (1,2,4,8,16):
            # Warm both endpoints with unmeasured, distinct prompts.
            for p in ports:stream(p,'Explain binary search in detail.',tokens=128,seed=100+repeat)
            count=conc*4;start=time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                futures=[pool.submit(stream,ports[i%len(ports)],f'Work item {i}. Write a detailed technical article: {TOPICS[i%len(TOPICS)]}. Include complete Python code, tests, and reasoning about tradeoffs. Continue until the topic is fully covered.',tokens=1024,seed=731+i) for i in range(count)]
                rows=[f.result() for f in futures]
            end=time.time();cell={'trial':repeat+1,'concurrency':conc,'summary':summary(rows,start,end),'requests':rows}
            record['cells'].append(cell);save(record['_path'],record)
            print(json.dumps({'concurrency':conc,'trial':repeat+1,**cell['summary']}),flush=True)


def interference(ports,record):
    decode_port=ports[0];prefill_port=ports[1] if len(ports)>1 else ports[0]
    record['definition']='Three paired idle vs 128K cold-prefill trials; decode already streaming before prefill is submitted. Gaps are stream-chunk gaps, not token timings.'
    record['trials']=[]
    for trial in range(3):
        for collision in (False,True):
            prompt,count=corpus(prefill_port,131072,'collision-'+uuid.uuid4().hex)
            began=threading.Event()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                decode=pool.submit(stream,decode_port,'Write a very detailed systems-programming textbook chapter on distributed transactions, including algorithms and code.',tokens=8192,seed=781+trial,first_event=began,timeout=300)
                if not began.wait(60):raise RuntimeError('Decode did not begin; collision was not exercised')
                time.sleep(1)
                prefill=pool.submit(stream,prefill_port,prompt,tokens=1,seed=921+trial,timeout=300) if collision else None
                row=decode.result();ingest=prefill.result() if prefill else None
            overlap=bool(ingest and row['started']<ingest['started']<row['ended'])
            out={'trial':trial+1,'collision':collision,'decode':row,'prefill':ingest,'prefill_prompt_tokens_targeted':count,'overlap_exercised':overlap}
            record['trials'].append(out);save(record['_path'],record)
            print('interference',trial+1,collision,'overlap',overlap,'max chunk gap',row['max_inter_chunk_gap_s'],flush=True)


def long_context(ports,record):
    port=ports[0]
    with session() as s:
        response=s.get(f'http://127.0.0.1:{port}/v1/models',timeout=10);response.raise_for_status();maxlen=response.json()['data'][0]['max_model_len']
    targets=[8192,131072,524288,770000]
    if maxlen>800000:targets.append(maxlen-8192)
    record['max_model_len']=maxlen;record['requests']=[]
    for size in targets:
        prompt,count=corpus(port,size,'retrieval-'+uuid.uuid4().hex)
        row=stream(port,prompt,tokens=4096,chat=True,timeout=1800)
        row.update(target_tokens=size,tokenized_prompt_tokens=count,alpha='84291' in row['content'],omega='FERN-72' in row['content'])
        row['retrieval_pass']=row['ok'] and row['alpha'] and row['omega']
        record['requests'].append(row);save(record['_path'],record)
        print('retrieval',size,count,row['retrieval_pass'],row.get('error'),flush=True)
    pressure(ports,record)


def pressure(ports,record):
    port=ports[0];prompts=[corpus(port,t,'pressure-'+uuid.uuid4().hex) for t in (700000,420000)]
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(stream,port,p,tokens=4096,chat=True,timeout=1800) for p,n in prompts]
        rows=[f.result() for f in fs]
    for row,(_,count) in zip(rows,prompts):
        row.update(tokenized_prompt_tokens=count,retrieval_pass=row['ok'] and '84291' in row['content'] and 'FERN-72' in row['content'])
    record['pressure']={'requested_prompt_tokens':sum(n for p,n in prompts),'summary':summary(rows,start,time.time()),'requests':rows}
    save(record['_path'],record);print('pressure',record['pressure']['summary'],flush=True)


def queue(ports,record):
    port=ports[0]
    # Distinct first blocks force genuinely cold prefills, not one shared cached prompt.
    prompts=[corpus(port,30000,'queue-'+str(i)+'-'+uuid.uuid4().hex) for i in range(12)]
    barrier=threading.Barrier(12)
    def request(i,p):
        barrier.wait(timeout=20)
        return stream(port,p,tokens=32,seed=731+i,timeout=240)
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        fs=[pool.submit(request,i,p) for i,(p,n) in enumerate(prompts)]
        rows=[f.result() for f in fs]
    record['queue']={'definition':'12 distinct cold 30K prompts released together into max_num_seqs=8; compare default max_parallel_prefills=1 control and explicit =2 stress.', 'summary':summary(rows,start,time.time()),'requests':rows}
    save(record['_path'],record);print('queue',record['queue']['summary'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['throughput','interference','long-context','pressure','queue'],required=True);ap.add_argument('--ports',required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise RuntimeError('Refusing to overwrite scenario evidence')
    ports=list(map(int,args.ports.split(',')));record={'schema':'lavd-scenarios/v1','mode':args.mode,'ports':ports,'started':time.time(),'_path':str(args.output),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    modes={'throughput':throughput,'interference':interference,'long-context':long_context,'pressure':pressure,'queue':queue}
    try:modes[args.mode](ports,record)
    except Exception as exc:record['error']=repr(exc);raise
    finally:record['ended']=time.time();save(args.output,record)

if __name__=='__main__':main()
