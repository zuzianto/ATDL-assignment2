#!/usr/bin/env python3
"""UCloud orchestration of authors' SUGFW+ Liver implementation. Run from any directory."""
import argparse, csv, os, subprocess, sys, json, pathlib
ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT/'data/Liver/Foreslices'
CKPT=ROOT/'checkpoint/sam_vit_b_01ec64.pth'
def run(args, cwd):
    print('RUN', ' '.join(map(str,args)), 'IN',cwd,flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preprocess','uncertainty','select','train','evaluate','inspect'])
    p.add_argument('--ratio',type=float,default=0.0005)
    p.add_argument('--scale',type=int,default=5)
    p.add_argument('--aug',type=int,default=10)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--selected-csv',type=pathlib.Path)
    p.add_argument('--checkpoint',type=pathlib.Path)
    a=p.parse_args()
    train=DATA/'splits/train.csv'
    if a.stage=='preprocess':
        raw=ROOT/'data/Liver/raw/Task03_Liver/imagesTr'
        if not raw.exists(): p.error(f'Missing dataset directory {raw}')
        run([sys.executable,'Liver_pre.py'],ROOT/'preprocessing')
        run([sys.executable,'save_spacing.py'],ROOT/'preprocessing')
        run([sys.executable,'Liver_csv.py'],ROOT/'preprocessing')
    elif a.stage=='inspect':
        for name in ['train','valid','test']:
            path=DATA/'splits'/f'{name}.csv'
            if path.exists():
                with path.open() as f: rows=list(csv.DictReader(f))
                cases={pathlib.Path(r['image_pth']).name.split('_')[0] for r in rows}
                print(name,'slices',len(rows),'cases',len(cases))
        if train.exists():
            with train.open() as f: n=sum(1 for _ in csv.DictReader(f))
            print('ratio',a.ratio,'selected_count',int(n*a.ratio))
    elif a.stage=='uncertainty':
        if not CKPT.exists(): p.error(f'Missing SAM checkpoint: {CKPT}')
        if not train.exists(): p.error('Preprocess first')
        run([sys.executable,'get_uncertainty.py','--model_type','vit_b','--checkpoint',CKPT,'--csv_path',train,'--k_times_aug',a.aug],ROOT/'UC-SAM')
    elif a.stage=='select':
        if not train.exists(): p.error('Preprocess first')
        code='from select_samples import Sample_Selector; import sys; s=Sample_Selector(sys.argv[1], float(sys.argv[2])); print("Selected count:",s.select_num); assert s.select_num>0, "Annotation budget rounds to zero"; s.sugfw_select(soft_scale=int(sys.argv[3]), un=int(sys.argv[4]))'
        run([sys.executable,'-c',code,train,a.ratio,a.scale,a.aug],ROOT/'UC-SAM')
    else:
        if a.stage=='train':
            if a.selected_csv: selected=a.selected_csv.resolve()
            else:
                if not train.exists(): p.error('Preprocess first')
                with train.open() as f: n=sum(1 for _ in csv.DictReader(f))
                selected=DATA/'splits'/f'my_select_{int(n*a.ratio)}_scale{a.scale}_un{a.aug}.csv'
            if not selected.exists(): p.error(f'Missing selected CSV: {selected}')
            out=ROOT/'output/Liver'/f'ratio{a.ratio}_scale{a.scale}_seed{a.seed}'
            out.mkdir(parents=True,exist_ok=True)
            (out/'run_config.json').write_text(json.dumps(vars(a),default=str,indent=2))
            run([sys.executable,'train.py','-b',a.batch_size,'--n_gpu',1,'--root_path',DATA,'--csv_path',selected,'-o',out,'--seed',a.seed,'--train_max_epochs',a.epochs,'--ckpt',CKPT],ROOT/'MA-SAM-un')
        else:
            ckpt=a.checkpoint.resolve() if a.checkpoint else None
            if not ckpt or not ckpt.exists(): p.error('Provide --checkpoint path/to/train_best_model.pth')
            run([sys.executable,'inference.py','--adapt_ckpt',ckpt,'--root_path',DATA,'--stage','test','--num_classes',2,'--img_size',512,'--seed',a.seed,'--cal_hd95','--ckpt',CKPT],ROOT/'MA-SAM-un')
if __name__=='__main__': main()
