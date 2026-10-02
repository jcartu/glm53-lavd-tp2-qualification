#!/usr/bin/env python3
"""Lavd's pinned TP2 qualification; reuse the field-lab ownership/restore guard."""
from __future__ import annotations
import argparse, copy, datetime, fcntl, hashlib, json, os, re, signal, subprocess, sys, threading, time
from pathlib import Path
import psutil, requests, yaml

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / 'manifest.json').read_text())
NAME = 'lavd-qualification'
LABEL = ROOT.name
BASE = 'http://127.0.0.1:8002'
BENCH = ROOT / 'preflight/llm_decode_bench.py'
PY = '/home/josh/llm-inference-bench/.venv/bin/python'
COMMON_ENV = {**os.environ, 'TZ': 'Europe/Berlin', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'https_proxy': 'http://127.0.0.1:9', 'http_proxy': 'http://127.0.0.1:9', 'NO_PROXY': 'localhost,127.0.0.1', 'no_proxy': 'localhost,127.0.0.1'}
HTTP = requests.Session(); HTTP.trust_env = False
os.environ.update(BATTERY_ROOT=str(ROOT), BATTERY_CONTAINER=NAME, BATTERY_PORT='8002', GPU_ISOLATION_MODE='strict')
sys.path.insert(0, '/home/josh/omp-workspace/glm53-flash-field-lab/scripts/r26')
import runtime as rt
import run_qualification as coordinator
coordinator.PRODUCTION = 'orca-prod'
rt.MODEL_NAME = 'g53fq'


def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp'); tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)
def note(text): print(stamp(), text, flush=True)
def command(argv, out, timeout=3600, env=None):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    note('RUN ' + out.stem)
    with out.open('w') as log:
        process = subprocess.Popen(list(map(str, argv)), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env or COMMON_ENV, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            if isinstance(exc, subprocess.TimeoutExpired): code = 124
            else: raise
    save(out.with_suffix('.command.json'), {'argv': list(map(str, argv)), 'exit_code': code, 'start': start, 'end': time.time(), 'elapsed_s': time.time()-start})
    note(f'END {out.stem}: {code}')
    return code


def owned():
    ids = subprocess.check_output(['docker','ps','-aq','--filter',f'label=lavd.campaign={LABEL}'],text=True).split()
    return json.loads(subprocess.check_output(['docker','inspect',*ids],text=True)) if ids else []


def stop_owned():
    for obj in owned():
        if obj['Config'].get('Labels',{}).get('lavd.campaign') != LABEL: raise RuntimeError('Container ownership changed')
        dest = ROOT/'arms'/obj['Config']['Labels'].get('lavd.arm','unclassified'); dest.mkdir(parents=True,exist_ok=True)
        stem = obj['Name'].strip('/')
        logs = subprocess.run(['docker','logs',obj['Id']],capture_output=True,text=True,timeout=45)
        (dest/(stem+'-final.log')).write_text(logs.stdout+logs.stderr)
        save(dest/(stem+'-inspect.json'),obj)
        subprocess.run(['docker','stop','--time','60',obj['Id']],check=True,timeout=90)
        subprocess.run(['docker','rm',obj['Id']],check=True,timeout=30)

# The shared guard supports all containers carrying field-lab.battery=r26.
# Replace its single-container cleanup so the dual-island arm is restored safely too.
rt.stop = stop_owned


def snapshot(dest, ports=(8002,)):
    dest=Path(dest); dest.mkdir(parents=True,exist_ok=True)
    for port in ports:
        for suffix in ('metrics','v1/models','version'):
            try:
                r=HTTP.get(f'http://127.0.0.1:{port}/{suffix}',timeout=10)
                (dest/(str(port)+'-'+suffix.replace('/','-')+'.txt')).write_text(r.text)
            except requests.RequestException as exc:
                (dest/(str(port)+'-'+suffix.replace('/','-')+'.error')).write_text(str(exc))
    for key,argv in [('gpu',['nvidia-smi','--query-gpu=index,uuid,pcie.link.gen.current,pcie.link.width.current,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem','--format=csv']),('memory',['free','-b'])]:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=20); (dest/(key+'.txt')).write_text(p.stdout+p.stderr)


def smoke(port=8002, model='g53fq'):
    body={'model':model,'messages':[{'role':'user','content':'What is 19 plus 23? Answer with only the number.'}],'max_tokens':2048,'temperature':0,'chat_template_kwargs':{'reasoning_effort':'low'}}
    response=HTTP.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body,timeout=180)
    response.raise_for_status(); data=response.json()
    text=data['choices'][0]['message'].get('content') or ''
    return {'ok':bool(re.search(r'\b42\b',text)), 'response':data,'port':port,'timestamp':stamp()}


def setarg(args,key,value):
    args[:]=[a for a in args if not a.startswith(key+'=')]
    args.append(key+'='+str(value))


def compose_for(arm):
    doc=yaml.safe_load((ROOT/'lavd-local.yaml').read_text()); svc=doc['services']['g53f']; args=svc['command']
    svc['labels']['lavd.arm']=arm
    env=dict(x.split('=',1) for x in svc['environment']); svc['environment']=env
    if arm in ('drafter-marlin','drafter-b12x'): svc['image']=META['drafter_image']
    if arm in ('scheduler-fixed','scheduler-fixed-stress'): svc['image']=META['scheduler_image']
    if arm=='drafter-b12x':
        spec=json.loads(next(a.split('=',1)[1] for a in args if a.startswith('--speculative-config=')))
        spec['moe_backend']='b12x'; setarg(args,'--speculative-config',json.dumps(spec))
    if arm.startswith('nccl-'):
        channels={'nccl-4':4,'nccl-16':16}[arm]
        env.update(NCCL_MIN_NCHANNELS=str(channels),NCCL_MAX_NCHANNELS=str(channels),NCCL_BUFFSIZE='2097152')
    if arm=='context-786k': setarg(args,'--max-model-len',786432)
    if arm=='accuracy-off': env.update(VLLM_B12X_MOE_FP4_FORCE_A16='0',B12X_W4A16_FP32_TOPK_WEIGHTS='0')
    if arm.endswith('-stress'): setarg(args,'--max-parallel-prefills',2)
    if arm=='qad-tp4':
        env['CUDA_VISIBLE_DEVICES']='0,1,2,3'; setarg(args,'--tensor-parallel-size',4)
        setarg(args,'--decode-context-parallel-size',1); env['VLLM_B12X_MLA_CKV_GATHER']='0'
        setarg(args,'--max-num-seqs',16); setarg(args,'--max-cudagraph-capture-size',64)
        setarg(args,'--compilation-config',json.dumps({'cudagraph_mode':'FULL_AND_PIECEWISE','cudagraph_capture_sizes':[1,2,4,8,12,16,20,24,28,32,40,48,56,64]}))
        # TP4 has free memory; let vLLM size its pool instead of needlessly imposing the TP2 cap.
        args[:]=[x for x in args if not x.startswith('--kv-cache-memory-bytes=')]
        setarg(args,'--gpu-memory-utilization',0.93)
    if arm=='spark-tp2':
        svc['image']=META['lavd_image']; svc['entrypoint']=['/usr/local/bin/lil-entrypoint']
        svc['command']=['--preset','glm53-spark-tp2','--model','/model','--host','127.0.0.1','--port','8002','--served-model-name','g53fq','--default-chat-template-kwargs','{"reasoning_effort":"max","clear_thinking":true}']
        svc['environment']={'CUDA_VISIBLE_DEVICES':'2,3','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CACHE_MODE':'vram'}
        svc['volumes'][0]=META['spark_model']+':/model:ro'
    if arm=='dual-tp2':
        sibling=copy.deepcopy(svc); sibling['container_name']=NAME+'-b'; sibling['environment']['CUDA_VISIBLE_DEVICES']='0,1'
        setarg(sibling['command'],'--port',8003)
        sibling['volumes']=[META['qad_model']+':/model:ro',str(ROOT/'cache-b')+':/root/.cache:rw',str(ROOT/'cache-b')+':/cache:rw',str(ROOT)+':/evidence:rw']
        doc['services']['second']=sibling
    return doc


def boot(arm):
    dest=ROOT/'arms'/arm; dest.mkdir(parents=True,exist_ok=True)
    doc=compose_for(arm); path=dest/'compose.yaml'; path.write_text(yaml.safe_dump(doc,sort_keys=False))
    for svc in doc['services'].values():
        for mount in svc['volumes'][1:3]: Path(mount.split(':',1)[0]).mkdir(parents=True,exist_ok=True)
    services=['g53f'] if arm=='dual-tp2' else []
    rc=command(['docker','compose','-p','lavd-qualification','-f',path,'up','-d',*services],dest/'launch.log',timeout=120)
    if rc: raise RuntimeError(f'Compose launch failed: {arm}')
    if arm=='dual-tp2':
        # Serial cold starts avoid overlapping host-memory peaks. Timed work starts only after both are warm.
        first_deadline=time.monotonic()+1800
        while time.monotonic()<first_deadline:
            first=owned()
            if not first or any(not x['State']['Running'] for x in first):
                raise RuntimeError('First island exited during startup')
            try:
                if HTTP.get(BASE+'/health',timeout=4).status_code==200: break
            except requests.RequestException: pass
            time.sleep(5)
        else: raise TimeoutError('First island startup timeout')
        rc=command(['docker','compose','-p','lavd-qualification','-f',path,'up','-d','second'],dest/'launch-second.log',timeout=120)
        if rc: raise RuntimeError('Second island launch failed')
    ports=(8002,8003) if arm=='dual-tp2' else (8002,)
    deadline=time.monotonic()+1800
    while time.monotonic()<deadline:
        objs=owned()
        if len(objs)!=len(ports) or any(not x['State']['Running'] for x in objs):
            raise RuntimeError('Worker exited before readiness')
        try:
            if all(HTTP.get(f'http://127.0.0.1:{p}/health',timeout=4).status_code==200 for p in ports):
                for p in ports:
                    check=smoke(p); save(dest/f'smoke-{p}.json',check)
                    if not check['ok']: raise RuntimeError('Real completion arithmetic gate failed')
                for obj in objs:
                    logs=subprocess.run(['docker','logs',obj['Id']],capture_output=True,text=True,timeout=30)
                    text=logs.stdout+logs.stderr; (dest/(obj['Name'].strip('/')+'-boot.log')).write_text(text)
                    pools=re.findall(r'GPU KV cache size:\s*([\d,]+)',text)
                    save(dest/(obj['Name'].strip('/')+'-capacity.json'),{'kv_tokens':[int(x.replace(',','')) for x in pools],'capacity_lines':[s for s in text.splitlines() if 'concurrency' in s or 'KV cache size' in s],'checkpoint':doc['services']['g53f']['volumes'][0], 'image':obj['Image']})
                snapshot(dest/'ready',ports);return ports
        except requests.RequestException: pass
        time.sleep(5)
    raise TimeoutError('Serving readiness timeout')


def bench(arm, tag, extra, timeout=7200, port=8002, model='g53fq', dcp=2):
    dest=ROOT/'arms'/arm; out=dest/(tag+'.json')
    if out.exists(): raise RuntimeError('Refusing to overwrite existing benchmark '+str(out))
    argv=[PY,str(BENCH),'--port',str(port),'--model',model,'--display-mode','plain','--no-resume','--output',str(out),'--dcp-size',str(dcp),*extra]
    snapshot(dest/(tag+'-before'),(port,))
    rc=command(argv,dest/(tag+'.log'),timeout)
    snapshot(dest/(tag+'-after'),(port,))
    return rc


def speed(arm, port=8002, model='g53fq', dcp=2):
    for i in range(3):
        bench(arm,f'decode-{i+1}', ['--concurrency','1,4,8','--contexts','0,16k','--duration','30','--max-tokens','8192','--temperature','1.0','--skip-prefill'],port=port,model=model,dcp=dcp)
    bench(arm,'prefill',['--prefill-only','--prefill-contexts','64k,128k','--prefill-duration','20','--token-targeting','exact'],port=port,model=model,dcp=dcp)
    bench(arm,'burst',['--skip-prefill','--concurrency','1,4,8','--contexts','0,16k','--request-count','40','--warmup-request-count','4','--max-tokens','512','--temperature','1.0'],port=port,model=model,dcp=dcp)


def profile(arm, name, count, max_tokens, port=8002, model='g53fq',dcp=2, tag=None):
    return bench(arm,tag or name+str(count),['--test-profile',name,'--profile-concurrency','8','--profile-runs',str(count),'--max-tokens',str(max_tokens),'--reasoning-effort','max','--completion-stats-seed','731','--completion-stats-request-timeout','3600','--completion-stats-save-text'],timeout=86400,port=port,model=model,dcp=dcp)


def lil(arm):
    dest=ROOT/'arms'/arm
    before={str(p) for p in (ROOT/'cache'/'lil-bench').glob('*.json.gz')}
    rc=command(['docker','exec','--privileged',NAME,'/opt/venv/bin/lil-bench','run','--no-upload','--profile','standard','--note',f'JCartu Lavd qualification: {arm}, isolated GPU run, original results retained'],dest/'lil-bench.log',timeout=14400)
    after=[str(p) for p in (ROOT/'cache'/'lil-bench').glob('*.json.gz') if str(p) not in before]
    save(dest/'lil-bench-files.json',{'returncode':rc,'files':after,'upload':'not attempted; no LIL_BENCH_TOKEN present'})


def scenarios(arm, mode, ports):
    return command(['/usr/bin/python',ROOT/'scenarios.py','--mode',mode,'--ports',','.join(map(str,ports)),'--output',ROOT/'arms'/arm/(mode+'.json')],ROOT/'arms'/arm/(mode+'.log'),timeout=14400)


def phase(arm):
    dest=ROOT/'arms'/arm; dest.mkdir(parents=True,exist_ok=True)
    save(dest/'driver-identity.json', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'scenarios.py',BENCH)})
    start=time.time(); stop=threading.Event()
    serving=threading.Event()
    def telemetry():
        last_activity=time.monotonic()
        with (dest/'telemetry.jsonl').open('a') as log:
            while not stop.wait(2):
                try:
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu,pcie.link.gen.current,clocks.sm,clocks.mem','--format=csv,noheader,nounits'],text=True,timeout=10)
                    if any(float(row.split(',')[1].strip()) > 0 for row in gpu.splitlines() if row.strip()):
                        last_activity=time.monotonic()
                    if not serving.is_set(): last_activity=time.monotonic()
                    if serving.is_set() and time.monotonic()-last_activity > 900:
                        save(dest/'idle-watchdog.json',{'error':'No GPU activity for 900 seconds after readiness','timestamp':stamp()})
                        os.kill(os.getpid(),signal.SIGINT)
                        return
                    metrics={}
                    for p in ((8002,8003) if arm=='dual-tp2' else (8002,)):
                        try:
                            text=requests.get(f'http://127.0.0.1:{p}/metrics',timeout=3).text
                            metrics[str(p)]='\n'.join(line for line in text.splitlines() if line.startswith('vllm:') and any(k in line for k in ('num_requests_','generation_tokens_total','prompt_tokens_total','preempt','cache_usage','spec_decode','iteration_tokens_total','request_success_total')))
                        except requests.RequestException: pass
                    log.write(json.dumps({'timestamp':time.time(),'gpus':gpu,'host_memory':psutil.virtual_memory()._asdict(),'host_swap':psutil.swap_memory()._asdict(),'metrics':metrics})+'\n');log.flush()
                except Exception as exc: log.write(json.dumps({'error':repr(exc),'timestamp':time.time()})+'\n');log.flush()
    monitor=threading.Thread(target=telemetry,daemon=True); monitor.start()
    try:
        ports=boot(arm)
        serving.set()
        subprocess.run(['tmux','wait-for','-S',LABEL+'-'+arm+'-ready'],check=True,timeout=10)
        if arm in ('reference','qad-tp4','spark-tp2'):
            lil(arm);speed(arm,dcp=1 if arm=='qad-tp4' else 2)
            profile(arm,'lavd-test',10,0,dcp=1 if arm=='qad-tp4' else 2)
            profile(arm,'mmlu-pro',200,131072,dcp=1 if arm=='qad-tp4' else 2)
            if arm=='reference':
                scenarios(arm,'throughput',ports);scenarios(arm,'interference',ports)
                profile(arm,'mmlu-pro',200,131072,tag='mmlu-pro200-repeat')
                profile(arm,'estonia',30,40000)
                profile(arm,'mmlu-pro',1000,131072)
                scenarios(arm,'long-context',ports)
                scenarios(arm,'queue',ports)
            if arm=='qad-tp4': scenarios(arm,'throughput',ports);scenarios(arm,'interference',ports)
        elif arm in ('drafter-marlin','drafter-b12x','accuracy-off'):
            speed(arm);profile(arm,'lavd-test',10,0)
            if arm=='accuracy-off': profile(arm,'mmlu-pro',200,131072)
        elif arm in ('nccl-4','nccl-16'):
            speed(arm)
        elif arm=='context-786k':
            scenarios(arm,'long-context',ports)
        elif arm.startswith('scheduler-'):
            scenarios(arm,'queue',ports)
        elif arm=='dual-tp2':
            scenarios(arm,'throughput',ports);scenarios(arm,'interference',ports)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                checks=[pool.submit(profile,arm+'-'+str(p),'lavd-test',10,0,p) for p in ports]
                for check in checks: check.result()
            for p in ports:
                check=smoke(p);save(dest/f'dual-final-quality-{p}.json',check)
        save(dest/'phase-complete.json',{'arm':arm,'started':start,'finished':time.time(),'status':'executed','note':'Individual benchmark scores and failures are retained; executed does not mean passed.'})
    except BaseException as exc:
        save(dest/'phase-error.json',{'arm':arm,'error':repr(exc),'timestamp':time.time()})
        raise
    finally:
        stop.set();monitor.join(timeout=20);stop_owned()


ARMS=['reference','qad-tp4','spark-tp2','drafter-marlin','drafter-b12x','nccl-4','nccl-16','context-786k','scheduler-fixed','scheduler-old-stress','scheduler-fixed-stress','accuracy-off','dual-tp2']
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=ARMS);parser.add_argument('--arms',default=','.join(ARMS));parser.add_argument('--production-baseline',action='store_true');parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    if args.plan_only:
        for arm in ARMS: save(ROOT/'preflight'/f'planned-{arm}.json',compose_for(arm))
        print(json.dumps({'arms':ARMS,'original_production':META['production_id']},indent=2));return
    if args.phase: phase(args.phase);return
    selected=args.arms.split(',')
    if any(a not in ARMS for a in selected): raise ValueError('Unknown arm')
    with (ROOT.parent/'lavd-gpu-campaign.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.production_baseline:
            if not coordinator.production_idle(): raise RuntimeError('Production busy')
            dest=ROOT/'arms'/'production-nvidia-tp4'; dest.mkdir(parents=True,exist_ok=True)
            check=smoke(5001,'GLM-5.3-Flash-NVFP4');save(dest/'smoke.json',check)
            if not check['ok']: raise RuntimeError('Production completion failed')
            speed('production-nvidia-tp4',5001,'GLM-5.3-Flash-NVFP4',1)
            profile('production-nvidia-tp4','lavd-test',10,0,5001,'GLM-5.3-Flash-NVFP4',1)
            profile('production-nvidia-tp4','mmlu-pro',200,131072,5001,'GLM-5.3-Flash-NVFP4',1)
            save(dest/'phase-complete.json',{'status':'executed','checkpoint':META['production_model']});return
        control=ROOT/'control'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        control.mkdir(parents=True,exist_ok=False)
        rt.ROOT=control
        coordinator.PHASES=[(a,str(Path(__file__)),['--phase',a],172800) for a in selected]
        try: coordinator.main()
        finally:
            restored=HTTP.get('http://127.0.0.1:5001/health',timeout=10)
            if restored.ok: save(rt.ROOT/'production-restored-completion.json',smoke(5001,'GLM-5.3-Flash-NVFP4'))

if __name__=='__main__': main()
