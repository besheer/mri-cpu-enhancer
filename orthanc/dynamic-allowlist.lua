local internalToken = os.getenv('INTERNAL_TOKEN') or ''
local calledAet = string.upper(os.getenv('ORTHANC_CALLED_AET') or 'MRI03T')

function ReceivedCStoreInstanceFilter(dicom, origin, info)
  local remoteIp = origin['RemoteIp'] or ''
  local remoteAet = string.lower(origin['RemoteAet'] or '')
  local requestedAet = string.upper(origin['CalledAet'] or '')
  local modality = dicom['Modality'] or ''
  local description = string.upper(dicom['SeriesDescription'] or '')

  -- Loop prevention: never accept series already processed by RAD4AI
  if string.sub(description, 1, 3) == 'ENH' or string.find(description, '%[RAD4AI') or string.find(description, '%[ENHANCED%]') or string.find(description, 'SYNTH%-') then
    return 0xA801
  end

  -- Whitelist primary modality (192.168.0.206 / adw) unconditionally with zero latency
  if remoteIp == '192.168.0.206' or remoteAet == 'adw' then
    return 0x0000
  end

  if modality ~= 'MR' then
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
