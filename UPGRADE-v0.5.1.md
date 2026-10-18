# Hotfix v0.5.1

يصحح التحقق من Orthanc Series ID في زر Process BASE. لا توجد تغييرات على البيانات أو Docker volumes.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.5.1.tar.gz
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
```
