import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
records=json.loads((root/'MANIFEST-SHA256.json').read_text(encoding='utf-8'))
for rel,expected in records.items():
    p=(root/rel).resolve();assert root in p.parents
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,rel
print(f'Verified {len(records)} release files')
