"""Select and hash-check original image bytes; no labels inferred or images fetched."""
import argparse,csv,hashlib,json,shutil,tarfile
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--splits',nargs='+',required=True);s=p.add_mutually_exclusive_group(required=True);s.add_argument('--images-root');s.add_argument('--archive');p.add_argument('--output',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    rows=[]
    for file in a.splits:
        with open(file,encoding='utf-8-sig',newline='') as f:
            rows += [dict(r,split_name=Path(file).stem) for r in csv.DictReader(f)]
    required={r['image_sha256']:r for r in rows};dest=Path(a.output).resolve();dest.mkdir(parents=True,exist_ok=False);images=dest/'images';images.mkdir()
    if a.archive:
        provenance=json.loads((Path(__file__).resolve().parents[1]/'splits/source-provenance.json').read_text(encoding='utf-8'))
        assert digest(a.archive)==provenance['image_archive']['sha256'],'Wrong archive bytes'
        membermap={r['archive_member']:h for h,r in required.items()}
        with tarfile.open(a.archive,'r|gz') as tar:
            for member in tar:
                if member.name not in membermap:continue
                if not member.isfile():raise ValueError('Expected regular image file')
                # Never extract paths from the archive. Destination is controlled SHA filename.
                h=membermap[member.name];target=images/(h+'.jpg')
                with tar.extractfile(member) as src,target.open('xb') as out:shutil.copyfileobj(src,out)
                assert digest(target)==h
        found={h:images/(h+'.jpg') for h in required if (images/(h+'.jpg')).exists()}
    else:
        names={Path(r['image_id']).name for r in rows};found={}
        for path in Path(a.images_root).rglob('*'):
            if path.is_file() and (path.name in names or path.stem in required):
                h=digest(path)
                if h in required:found.setdefault(h,path)
    missing=sorted(set(required)-set(found))
    if missing:raise ValueError(f'{len(missing)} required images missing; first {missing[:3]}')
    for h,source in found.items():
        target=images/(h+'.jpg')
        if not a.dry_run and source.resolve()!=target.resolve():shutil.copyfile(source,target);assert digest(target)==h
    manifest=[dict(r,image_path=str(found[r['image_sha256']].resolve()) if a.dry_run else str(images/(r['image_sha256']+'.jpg'))) for r in rows]
    with (dest/'manifest.json').open('x',encoding='utf-8') as f:json.dump(dict(images=manifest,dry_run=a.dry_run,unique_images=len(required)),f,indent=2)
    print(json.dumps(dict(rows=len(rows),unique_images=len(required),hashes_verified=len(found),dry_run=a.dry_run)))
if __name__=='__main__':main()
