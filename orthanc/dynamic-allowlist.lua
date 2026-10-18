local internalToken = os.getenv('INTERNAL_TOKEN') or ''
local calledAet = os.getenv('ORTHANC_CALLED_AET') or 'MRI03T'

function ReceivedCStoreInstanceFilter(dicom, origin, info)
  local remoteIp = origin['RemoteIp'] or ''
  local remoteAet = origin['RemoteAet'] or ''
  local requestedAet = origin['CalledAet'] or ''
  local modality = dicom['Modality'] or ''
  local description = string.upper(dicom['SeriesDescription'] or '')

  if requestedAet ~= calledAet or modality ~= 'MR' or string.sub(description, 1, 7) == 'ENHANC-' then
    return 0xA801
  end

  local headers = { ['X-Internal-Token'] = internalToken }
  local url = 'http://enhancer:8000/internal/dicom-authorize?ip=' .. remoteIp .. '&aet=' .. remoteAet .. '&called_aet=' .. requestedAet
  local ok, result = pcall(HttpGet, url, headers)
  if ok and result == 'ALLOW' then
    return 0x0000
  end
  print('Rejected C-STORE: IP=' .. remoteIp .. ' AET=' .. remoteAet)
  return 0xA801
end

function OnStoredInstance(instanceId, tags, metadata, origin)
  if origin['RequestOrigin'] == 'DicomProtocol' then
    RestApiPut('/instances/' .. instanceId .. '/metadata/1024', origin['RemoteAet'] or '')
  end
end
