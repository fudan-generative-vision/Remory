"""Publish an immutable, audited selection of events with validated cached summaries.

The source dataset and resumable source cache are preserved. Selection is explicit:
a trainer must supply the resulting selection manifest to use partial coverage.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
from pathlib import Path
from context_compression.prepare_qwen38_residual_summaries import validate_dataset
from context_compression.qwen38_allturn_data import digest
from context_compression.qwen38_residual_data import SummaryCache, identity

SCHEMA = "qwen38_residual_qualified_summary_selection_v1"

def load_selection(path, dataset_manifest, summary_manifest_path):
    value=json.loads(Path(path).read_text());body={k:v for k,v in value.items() if k!="selection_sha256"}
    if value.get("schema")!=SCHEMA or identity(body)!=value.get("selection_sha256"):
        raise ValueError("selection identity mismatch")
    if value["database_sha256"]!=dataset_manifest["database_sha256"] or value["summary_manifest_sha256"]!=digest(Path(summary_manifest_path)):
        raise ValueError("selection data/summary identity mismatch")
    ordinals=value["ordinals"]
    if not ordinals or any(type(n) is not int or n<0 for n in ordinals) or ordinals!=sorted(set(ordinals)):
        raise ValueError("invalid selected ordinals")
    excluded=value["excluded"]
    total=sum(v["trajectories"] for v in dataset_manifest["stats"].values())
    missing=[r["ordinal"] for r in excluded]
    if (len(set(missing))!=len(missing) or set(ordinals)&set(missing) or set(ordinals)|set(missing)!=set(range(total))
            or value["selected_responses"]!=len(ordinals) or value["original_responses"]!=total):
        raise ValueError("selection coverage mismatch")
    receipt={k:v for k,v in value.items() if k not in ("ordinals","excluded")}
    receipt["manifest"]=str(Path(path).resolve());receipt["manifest_sha256"]=digest(Path(path));receipt["excluded_responses"]=len(excluded)
    return ordinals,receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--target-model',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--failure-log',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    manifest,db=validate_dataset(a.manifest,a.target_model)
    if 'derivation' not in manifest or db.execute('select count(*) from examples where responses!=1').fetchone()[0]:raise ValueError('selection requires derived independent single-response events')
    output=a.output_dir.resolve();staging=output.with_name('.'+output.name+'.staging')
    if output.exists() or staging.exists():raise FileExistsError(output)
    staging.mkdir(parents=True)
    original=sqlite3.connect(f'file:{a.cache.resolve()}?mode=ro',uri=True);snapshot=sqlite3.connect(staging/'summaries.sqlite');original.backup(snapshot);original.close()
    contract=json.loads(snapshot.execute("select value from metadata where key='contract'").fetchone()[0]);snapshot.close()
    cache=SummaryCache(staging/'summaries.sqlite',contract)
    if cache.db.execute("select 1 from metadata where key='sealed'").fetchone():raise ValueError('source is already sealed; use its full manifest')
    failures={}
    for line in a.failure_log.read_text().splitlines():
        if line.strip():
            row=json.loads(line);failures[(row['key'],row['turn'])]=row
    selected=[];excluded=[];tokens=0
    for ordinal,key,target_tokens in db.execute('select ordinal,key,tokens from examples order by ordinal'):
        if cache.db.execute('select 1 from summaries where key=? and turn=0',(key,)).fetchone():selected.append(ordinal);tokens+=target_tokens
        else:excluded.append(dict(ordinal=ordinal,key=key,turn=0,reason=failures.get((key,0),{}).get('error','no validated cached summary at snapshot time')))
    print(json.dumps(dict(phase='sealing_selection',selected=len(selected),excluded=len(excluded))),flush=True)
    summary=cache.seal(db,ordinals=selected);cache.close();db.close();summary['cache']=str(output/'summaries.sqlite')
    (staging/'summary.manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    selection=dict(schema=SCHEMA,database_sha256=manifest['database_sha256'],source_manifest=str(a.manifest.resolve()),source_cache=str(a.cache.resolve()),summary_manifest_sha256=digest(staging/'summary.manifest.json'),policy='all_and_only_cached_validated_independent_events_at_snapshot',original_responses=len(selected)+len(excluded),selected_responses=len(selected),selected_target_tokens=tokens,ordinals=selected,excluded=excluded)
    selection['selection_sha256']=identity(selection);(staging/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    load_selection(staging/'selection.json',manifest,staging/'summary.manifest.json')
    if output.exists():raise FileExistsError(output)
    os.rename(staging,output);print(json.dumps(dict(output=str(output),selected=len(selected),excluded=len(excluded),target_tokens=tokens)),flush=True)

if __name__=='__main__':main()
