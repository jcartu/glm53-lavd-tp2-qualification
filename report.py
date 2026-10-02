#!/usr/bin/env python3
"""Render phone-legible figures only from completed, retained measurements."""
from __future__ import annotations
import argparse, collections, hashlib, importlib.util, json, math, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib import font_manager
from PIL import Image

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'charts';OUT.mkdir(exist_ok=True)
BG='#09131f';PANEL='#101e2d';INK='#f3f7fa';MUTED='#9dafbf';GRID='#233646'
TEAL='#58e0c2';BLUE='#6fafff';GOLD='#f3c568';PURPLE='#b5a0ff';RED='#fb8a8f'
LABELS={'reference':'Lavd · 2 GPUs','qad-tp4':'QAD · 4 GPUs','dual-tp2':'Two copies · 4 GPUs','spark-tp2':'Spark preset · 2 GPUs','production-nvidia-tp4':'NVIDIA · 4 GPUs','drafter-marlin':'Marlin','drafter-b12x':'B12X','nccl-4':'4 channels','nccl-16':'16 channels','accuracy-off':'A16 / FP32 knobs off'}
COLORS={'reference':TEAL,'qad-tp4':BLUE,'dual-tp2':GOLD,'spark-tp2':PURPLE,'production-nvidia-tp4':MUTED}
font_manager.fontManager.addfont(str(ROOT/'fonts/Inter-Regular.ttf'))
font_manager.fontManager.addfont(str(ROOT/'fonts/Inter-Bold.ttf'))
plt.rcParams.update({'font.family':'Inter','font.size':18,'text.color':INK,'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'axes.edgecolor':GRID,'figure.facecolor':BG,'savefig.facecolor':BG,'axes.facecolor':BG,'svg.fonttype':'none'})

def load(path):return json.loads(Path(path).read_text())
def med(xs):return statistics.median(xs) if xs else None

def valid(row):
    return row.get('aggregate_tps',0)>0 and not any(row.get(k) for k in ('num_errors','failure_reason','capacity_limited','loop_detected','warmup_timed_out','timeout_reason')) and not str(row.get('status','')).lower().startswith(('error','fail','skip'))

def collect():
    data={'schema':'lavd-public-summary/v1','arms':{},'source_files':{},'comparisons':{}}
    for arm in sorted((ROOT/'arms').iterdir()):
        if not arm.is_dir():continue
        entry={'decode':[],'profiles':{},'errors':[],'scenario_summaries':{}}
        for file in sorted(arm.glob('decode-*.json')):
            if '.command.' in file.name or '.resume.' in file.name:continue
            report=load(file);data['source_files'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            for row in report.get('results',[]):
                entry['decode'].append({'trial':file.stem,'concurrency':row.get('concurrency'),'context_tokens':row.get('context_tokens'),'aggregate_tps':row.get('aggregate_tps'),'valid':valid(row),'errors':row.get('num_errors'),'capacity_limited':row.get('capacity_limited'),'spec_acceptance':row.get('server_spec_accept_rate'),'spec_accept_length':row.get('server_spec_accept_length')})
        for file in sorted(arm.glob('*.json')):
            if file.name.startswith(('mmlu-pro','lavd-test','estonia')) and not any(x in file.name for x in ('.command.','.resume.')):
                report=load(file)
                if not report.get('metadata',{}).get('mode')=='completion_stats':continue
                entry['profiles'][file.stem]={'metadata':{k:v for k,v in report.get('metadata',{}).items() if k in ('test_profile','dataset_sha256','prompt_sha256','max_tokens','reasoning_effort','temperature','requested_runs','fixed_concurrency','interrupted')},'accuracy':report.get('accuracy'),'summary':report.get('all_summary')}
                data['source_files'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            if file.name=='prefill.json':
                report=load(file)
                entry['prefill']=report.get('prefill',{})
                data['source_files'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            if file.name.endswith('capacity.json'):entry['capacity']=load(file)
            if file.name=='phase-error.json':entry['errors'].append(load(file))
            if file.name in ('throughput.json','interference.json','long-context.json','pressure.json','queue.json'):
                src=load(file);key=file.stem
                data['source_files'][str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
                if key=='throughput':entry['scenario_summaries'][key]=[{k:v for k,v in c.items() if k!='requests'} for c in src.get('cells',[])]
                elif key=='interference':entry['scenario_summaries'][key]=[{'trial':t['trial'],'collision':t['collision'],'overlap_exercised':t['overlap_exercised'],'decode_ok':t['decode']['ok'],'decode_tpot_ms':t['decode']['tpot_ms'],'max_inter_chunk_gap_s':t['decode']['max_inter_chunk_gap_s']} for t in src.get('trials',[])]
                elif key=='queue':entry['scenario_summaries'][key]=src.get('queue',{}).get('summary',{})
                else:entry['scenario_summaries'][key]={'retrieval':[{'tokens':r['tokenized_prompt_tokens'],'pass':r['retrieval_pass'],'elapsed_s':r['elapsed_s'],'error':r.get('error')} for r in src.get('requests',[])],'pressure':src.get('pressure',{}).get('summary',{})}
        data['arms'][arm.name]=entry
    spec=importlib.util.spec_from_file_location('retained_bench',ROOT/'preflight/llm_decode_bench.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    base=ROOT/'arms/reference/mmlu-pro200.json'
    if base.exists():
        for arm,name in [('reference','mmlu-pro200-repeat'),('accuracy-off','mmlu-pro200'),('qad-tp4','mmlu-pro200'),('spark-tp2','mmlu-pro200')]:
            path=ROOT/'arms'/arm/(name+'.json')
            if path.exists():data['comparisons'][arm+'-'+name]=module.build_paired_comparison(load(base),load(path),'Lavd reference',arm)
    (ROOT/'public-summary.json').write_text(json.dumps(data,indent=2)+'\n')
    return data

def canvas(title,subtitle,footer,figsize=(14,10),kicker='GLM FLASH  /  QAD STEP 3500'):
    fig=plt.figure(figsize=figsize,facecolor=BG)
    fig.text(.065,.94,kicker,color=TEAL,size=16,weight='bold',va='top')
    fig.text(.065,.887,title,size=32,weight='bold',va='top')
    fig.text(.065,.826,subtitle,size=18,color=MUTED,va='top')
    fig.text(.065,.055,footer,size=14,color=MUTED,va='bottom',linespacing=1.6)
    return fig

def style(ax,xlabel='',ylabel=''):
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='y',color=GRID,alpha=.7,linewidth=.8,zorder=0)
    ax.tick_params(length=0,pad=10,labelsize=17);ax.set_xlabel(xlabel,fontsize=18,labelpad=14);ax.set_ylabel(ylabel,fontsize=18,labelpad=12)
    ax.yaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:,.0f}'))
    ax.set_axisbelow(True)

def export(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,facecolor=BG)
    fig.savefig(OUT/(name+'.svg'),facecolor=BG)
    plt.close(fig)
    with Image.open(OUT/(name+'.png')) as image:
        image.thumbnail((560,1000));image.convert('RGB').save(OUT/(name+'-phone.png'))
    return name+'.png'

def deployment(data):
    arms=[a for a in ('reference','qad-tp4','dual-tp2') if data['arms'].get(a,{}).get('scenario_summaries',{}).get('throughput')]
    if not arms:return None
    fig=canvas('Two cards. Four cards. Two copies.','Total generation throughput at the same request load.','Same QAD checkpoint + prompts · 1,024 output tokens per request\nMedian of clean trials; bands show range. Failed cells stay in the raw results.')
    ax=fig.add_axes([.105,.205,.72,.55]);style(ax,'Concurrent requests','Output tokens / second')
    for arm in arms:
        grouped=collections.defaultdict(list)
        for c in data['arms'][arm]['scenario_summaries']['throughput']:
            if c['summary']['failed']==0:grouped[c['concurrency']].append(c['summary']['aggregate_output_tokens_per_s'])
        xs=sorted(grouped);ys=[med(grouped[x]) for x in xs]
        if not xs:continue
        positions=[(1,2,4,8,16).index(x) for x in xs]
        ax.plot(positions,ys,color=COLORS[arm],lw=3.3,marker='o',ms=8,label=LABELS[arm],zorder=3)
        ax.fill_between(positions,[min(grouped[x]) for x in xs],[max(grouped[x]) for x in xs],color=COLORS[arm],alpha=.13)
        ax.annotate(f'{ys[-1]:,.0f}',(positions[-1],ys[-1]),xytext=(12,0),textcoords='offset points',color=COLORS[arm],fontsize=21,weight='bold',va='center')
    ax.set_xticks(range(5),['1','2','4','8','16']);ax.set_xlim(-.12,4.65);ax.set_ylim(bottom=0)
    ax.legend(loc='upper left',frameon=False,fontsize=17,labelcolor=INK)
    return export(fig,'01-two-cards-four-cards')

def reference_chart(data):
    arms=[a for a in ('reference','qad-tp4','spark-tp2','production-nvidia-tp4') if any(r['valid'] and r['context_tokens']==0 for r in data['arms'].get(a,{}).get('decode',[]))]
    if not arms:return None
    fig=canvas('Same box. Different setups.','Sustained decode on short prompts. Total across all active users.','Three 30-second trials; median and range of clean cells. Errors retained separately.\nDifferent checkpoints: these are serving comparisons, not knob-only A/Bs.',kicker='GLM FLASH  /  RTX PRO 6000 BLACKWELL')
    ax=fig.add_axes([.11,.23,.83,.48]);style(ax,'Concurrent requests','Output tokens / second')
    width=.76/max(len(arms),1)
    for i,arm in enumerate(arms):
        vals=[];lo=[];hi=[]
        for c in (1,4,8):
            xs=[r['aggregate_tps'] for r in data['arms'][arm]['decode'] if r['valid'] and r['context_tokens']==0 and r['concurrency']==c]
            value=med(xs);vals.append(value if value is not None else np.nan);lo.append(value-min(xs) if xs else 0);hi.append(max(xs)-value if xs else 0)
        xx=np.arange(3)+(i-(len(arms)-1)/2)*width
        bars=ax.bar(xx,vals,width*.88,color=COLORS[arm],label=LABELS[arm],zorder=3,yerr=[lo,hi],error_kw={'ecolor':INK,'capsize':3,'linewidth':1})
        for bar,value,up in zip(bars,vals,hi):
            if np.isfinite(value):ax.text(bar.get_x()+bar.get_width()/2,value+up+np.nanmax(vals)*.025,f'{value:,.0f}',ha='center',va='bottom',size=15,weight='bold',color=COLORS[arm])
    ax.set_xticks(range(3),['1','4','8']);ax.set_ylim(bottom=0);ax.margins(y=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.09,.785),ncol=2,frameon=False,fontsize=16,labelcolor=INK)
    return export(fig,'02-serving-setups')

def tuning(data):
    pairs=[('reference','nccl-4','NCCL: 4ch / 2 MiB'),('reference','nccl-16','NCCL: 16ch / 2 MiB'),('drafter-marlin','drafter-b12x','MTP: B12X vs Marlin'),('reference','accuracy-off','Accuracy knobs off')]
    rows=[]
    for base,cand,label in pairs:
        values=[]
        for c in (1,8):
            b=[r['aggregate_tps'] for r in data['arms'].get(base,{}).get('decode',[]) if r['valid'] and r['context_tokens']==0 and r['concurrency']==c]
            a=[r['aggregate_tps'] for r in data['arms'].get(cand,{}).get('decode',[]) if r['valid'] and r['context_tokens']==0 and r['concurrency']==c]
            values.append((med(a)/med(b)-1)*100 if a and b else None)
        if any(v is not None for v in values):rows.append((label,values))
    if not rows:return None
    fig=canvas('Which tweaks were worth it?','Measured decode change versus the matching control.','Same checkpoint and image within each A/B · median of 3 trials\nA speed gain is not a quality gain; accuracy results are reported separately.')
    ax=fig.add_axes([.35,.23,.55,.48]);style(ax,'Change in throughput (%)','');ax.grid(False);ax.grid(axis='x',color=GRID);ax.axvline(0,color=MUTED,lw=1)
    for i,(label,values) in enumerate(rows):
        for j,value in enumerate(values):
            if value is None:continue
            y=i+(j-.5)*.22;color=TEAL if j==0 else BLUE
            ax.scatter(value,y,s=105,color=color,zorder=3)
            ax.annotate(f'{value:+.1f}%',(value,y),xytext=(9 if value>=0 else -9,0),textcoords='offset points',va='center',ha='left' if value>=0 else 'right',color=color,size=18,weight='bold')
    ax.set_yticks(range(len(rows)),[r[0] for r in rows]);ax.invert_yaxis();ax.set_ylim(len(rows)-.5,-.55);ax.margins(x=.32)
    ax.plot([],[],color=TEAL,marker='o',ls='',label='1 request');ax.plot([],[],color=BLUE,marker='o',ls='',label='8 requests');ax.legend(loc='lower left',bbox_to_anchor=(-.45,1.04),ncol=2,frameon=False,labelcolor=INK,fontsize=17)
    return export(fig,'03-tuning-results')

def quality(data):
    arms=[a for a in ('reference','qad-tp4','spark-tp2','accuracy-off') if data['arms'].get(a,{}).get('profiles',{}).get('mmlu-pro200',{}).get('accuracy')]
    if not arms:return None
    fig=canvas('Same 200 questions. Same scoring.','MMLU-Pro, paired across the tested setups.','Whiskers: 95% Wilson intervals · greedy decoding, same prompt set\nThe unchanged reference moved 84% to 87% on repeat. Small deltas need caution.',kicker='GLM FLASH  /  PAIRED QUALITY CHECK')
    ax=fig.add_axes([.32,.25,.56,.47]);style(ax,'Correct answers (%)','');ax.grid(False);ax.grid(axis='x',color=GRID)
    for i,arm in enumerate(arms):
        ac=data['arms'][arm]['profiles']['mmlu-pro200']['accuracy'];v=ac['accuracy']*100;c=COLORS.get(arm,RED)
        ax.errorbar(v,i,xerr=[[v-ac['wilson95_low']*100],[ac['wilson95_high']*100-v]],fmt='o',markersize=10,color=c,elinewidth=2.4,capsize=5)
        ax.annotate(f"{ac['correct']}/{ac['scored']}",(v,i),xytext=(0,-28),textcoords='offset points',ha='center',size=19,color=c,weight='bold')
    ax.set_yticks(range(len(arms)),[LABELS[a] for a in arms]);ax.set_xlim(0,100);ax.set_ylim(len(arms)-.4,-.7);ax.spines['bottom'].set_color(GRID)
    return export(fig,'04-quality-check')

def interruption(data):
    arms=[a for a in ('reference','qad-tp4','dual-tp2') if data['arms'].get(a,{}).get('scenario_summaries',{}).get('interference')]
    if not arms:return None
    fig=canvas('Does one big prompt slow the other chat?','Longest streaming pause while a cold 128K prompt is ingested.','Three paired trials · median and range · lower is better\nTwo-copy setup sends the large prompt to the other instance. These are stream-chunk gaps.')
    ax=fig.add_axes([.12,.24,.81,.47]);style(ax,'','Longest stream pause (seconds)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'))
    for condition,color,offset,label in [(False,MUTED,-.16,'No competing prompt'),(True,TEAL,.16,'128K prompt arriving')]:
        vals=[];lower=[];upper=[]
        for arm in arms:
            rows=data['arms'][arm]['scenario_summaries']['interference']
            samples=[r['max_inter_chunk_gap_s'] for r in rows if r['collision']==condition and r['decode_ok'] and (not condition or r['overlap_exercised']) and r['max_inter_chunk_gap_s'] is not None]
            v=med(samples);vals.append(v if v is not None else np.nan);lower.append(v-min(samples) if samples else 0);upper.append(max(samples)-v if samples else 0)
        xx=np.arange(len(arms))+offset
        ax.bar(xx,vals,.27,color=color,label=label,yerr=[lower,upper],error_kw={'ecolor':INK,'capsize':4},zorder=3)
        for x,v,up in zip(xx,vals,upper):
            if np.isfinite(v):ax.annotate(f'{v:.2f}s',(x,v+up),xytext=(0,10),textcoords='offset points',ha='center',color=color,size=19,weight='bold')
    ax.set_xticks(range(len(arms)),[LABELS[a] for a in arms]);ax.set_ylim(bottom=0);ax.margins(y=.3)
    ax.legend(loc='lower left',bbox_to_anchor=(-.03,1.03),frameon=False,ncol=2,labelcolor=INK,fontsize=16)
    return export(fig,'05-prefill-interference')


def context_retrieval(data):
    rows=data['arms'].get('reference',{}).get('scenario_summaries',{}).get('long-context',{}).get('retrieval',[])
    if not rows:return None
    fig=canvas('Two GPUs. A million tokens.','Lavd’s recipe · cold prompts · two facts planted far apart.','Single cold request at each length; both facts placed near 5% and 95% depth.\nThis tests retrieval, not general long-context reasoning quality.')
    ax=fig.add_axes([.11,.23,.83,.5]);style(ax,'Prompt length (tokens)','Seconds to answer')
    xs=[r['tokens'] for r in rows];ys=[r['elapsed_s'] for r in rows]
    ax.plot(xs,ys,color=TEAL,lw=3,zorder=2)
    ax.fill_between(xs,ys,color=TEAL,alpha=.07)
    for r in rows:
        color=TEAL if r['pass'] else RED
        ax.scatter(r['tokens'],r['elapsed_s'],color=color,s=90,zorder=3)
        ax.annotate(f"{r['elapsed_s']:.1f}s",(r['tokens'],r['elapsed_s']),xytext=(0,13),textcoords='offset points',ha='center',color=color,size=20,weight='bold')
    ax.set_xticks([0,250000,500000,750000,1000000],['0','250K','500K','750K','1M'])
    ax.set_xlim(-30000,1120000);ax.set_ylim(0,max(ys)*1.2)
    ax.text(.035,.92,f"{sum(r['pass'] for r in rows)} / {len(rows)} lengths passed",transform=ax.transAxes,size=24,weight='bold',color=INK)
    ax.text(.035,.83,f"Longest: {max(xs):,} actual prompt tokens",transform=ax.transAxes,size=17,color=MUTED)
    return export(fig,'06-million-token-retrieval')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');parser.add_argument('--interim',action='store_true');args=parser.parse_args()
    data=collect()
    if args.interim:
        data['arms']={k:v for k,v in data['arms'].items() if k in ('reference','production-nvidia-tp4')}
        original_canvas=globals()['canvas']
        def interim_canvas(*a,**kw):
            fig=original_canvas(*a,**kw)
            fig.text(.94,.94,'INTERIM',ha='right',va='top',size=15,color=MUTED,weight='bold')
            return fig
        globals()['canvas']=interim_canvas
    if not args.partial and not args.interim:
        planned=load(ROOT/'execution-scope.json')['arms']
        unfinished=[arm for arm in planned if not any((ROOT/'arms'/arm/f).exists() for f in ('phase-complete.json','phase-error.json'))]
        if unfinished:raise RuntimeError('Incomplete arms: '+', '.join(unfinished))
    figures=[p for p in ((reference_chart(data),context_retrieval(data)) if args.interim else (deployment(data),reference_chart(data),tuning(data),quality(data),interruption(data),context_retrieval(data))) if p]
    (OUT/'manifest.json').write_text(json.dumps({'figures':figures,'partial':args.partial or args.interim,'interim':args.interim,'data_sha256':hashlib.sha256((ROOT/'public-summary.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print(json.dumps({'figures':figures,'arms':list(data['arms'])},indent=2))
if __name__=='__main__':main()
