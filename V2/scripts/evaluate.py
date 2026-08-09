import argparse,csv,json,statistics
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--logs',default='V2/results/logs'); p.add_argument('--output',default='V2/results/evaluation/summary.json'); a=p.parse_args(); rows=[]
for f in Path(a.logs).glob('*.csv'):
    with f.open() as s: rows += list(csv.DictReader(s))
def vals(k): return [float(r[k]) for r in rows if r.get(k,'')!='']
out={'frames':len(rows),'videos':len(list(Path(a.logs).glob('*.csv'))),'segmentation_confidence_mean':statistics.mean(vals('segmentation_confidence')) if rows else 0,'road_occupancy_mean':statistics.mean(vals('road_occupancy')) if rows else 0,'waypoint_correction_frequency':sum(abs(float(r['waypoint_x'])-float(r['corrected_x']))>.001 for r in rows)/max(1,len(rows)),'road_lost_events':sum('ROAD_LOST' in r['warning'] for r in rows),'recovery_triggers':sum(r['state'].startswith('RECOVERY') for r in rows),'steering_abs_mean':statistics.mean(abs(x) for x in vals('steering')) if rows else 0}
Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

