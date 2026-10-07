"""Download precisely the upstream source revision used by the demonstration."""
from pathlib import Path
import io, zipfile, urllib.request, shutil
ROOT=Path(__file__).resolve().parent
SHA='d47989841a86dfaf10b6cb3dbf98a43005429045'
target=ROOT/'upstream'
if target.exists():
    raise SystemExit('upstream already exists; retain it, or remove it explicitly before downloading.')
data=urllib.request.urlopen('https://codeload.github.com/tasyaa04/dynShap/zip/'+SHA,timeout=60).read()
z=zipfile.ZipFile(io.BytesIO(data))
prefix=z.namelist()[0].split('/')[0]
temp=ROOT/'upstream_download'
z.extractall(temp)
shutil.move(str(temp/prefix),target)
temp.rmdir()
print('Downloaded upstream revision',SHA)
