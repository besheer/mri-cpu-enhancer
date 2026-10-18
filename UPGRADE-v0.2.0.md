# Upgrade checklist: v0.1.0 to v0.2.0

1. Back up `/home/mri`, `.env`, and the Orthanc volume before upgrading.
2. Keep `AUTO_PROCESS_ENABLED=false` during installation.
3. Set long, unique values for `ORTHANC_PASSWORD`, `ADMIN_PASSWORD`, and `INTERNAL_TOKEN`.
4. Bind DICOM to a VPN/private address using `DICOM_BIND_IP`; never to `0.0.0.0` on the public server.
5. Start the containers and verify `/health`.
6. Open `/admin` and add the first center: allowed source IP/AET and return IP/AET/port.
7. Confirm a wrong IP or AE is rejected.
8. Run C-ECHO to the return destination and send a synthetic MR series.
9. Confirm that `ENHANC-...` returns to the test destination.
10. Only then set `AUTO_PROCESS_ENABLED=true`.

This remains research software and is not approved for diagnostic use.
