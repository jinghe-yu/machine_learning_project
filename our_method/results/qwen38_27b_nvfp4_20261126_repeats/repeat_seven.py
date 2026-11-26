from pathlib import Path
import json,subprocess,time
root=Path(__file__).resolve().parent
status={'status':'running','runs':{}}
def save():
 status['updated_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (root/'batch_status.json').write_text(json.dumps(status,indent=2)+'\n')
save()
for index in (2,3):
 name=f'run_{index:02d}'
 r=root/name
 status['runs'][name]={'status':'running'}
 save()
 with (r/'run.log').open('w') as log:
  process=subprocess.Popen(['/usr/bin/python3','-u','run_qwen_smoke.py','--factorial','--samples','56','--seed','42','--output-dir',str(r/'raw')],cwd=r/'our_method',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  status['runs'][name]['pid']=process.pid
  save()
  code=process.wait()
 status['runs'][name].update(status='complete' if code==0 else 'failed',exit_code=code)
 if code:
  status['status']='failed'
  save()
  raise SystemExit(code)
 save()
status['status']='complete'
save()
