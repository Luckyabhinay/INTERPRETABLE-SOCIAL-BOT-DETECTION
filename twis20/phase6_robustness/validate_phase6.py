import argparse,json
from pathlib import Path
def main():
 a=argparse.ArgumentParser();a.add_argument('output',type=Path);x=a.parse_args().output;d=json.loads((x/'validation.json').read_text(encoding='utf-8'));assert d['hashes_before']==d['hashes_after'];assert d['model_inputs']==d['feature_names']==50080;assert d['test_label_tuning'] is False;print('PASS',d['baseline']['accuracy'])
if __name__=='__main__':main()
