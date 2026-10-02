#!/usr/bin/env python3
"""Wait for a JSON result using Linux inotify, without polling files or processes."""
import ctypes, hashlib, json, os, struct, sys
from pathlib import Path
path=Path(sys.argv[1]).resolve()
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
try:
    if libc.inotify_add_watch(fd,os.fsencode(path.parent),0x8|0x80)<0:
        raise OSError(ctypes.get_errno(),'inotify_add_watch')
    while True:
        try:
            raw=path.read_bytes();data=json.loads(raw)
            print(json.dumps({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'keys':list(data)}),flush=True)
            break
        except (FileNotFoundError,json.JSONDecodeError):pass
        events=os.read(fd,65536)
        offset=0
        matched=False
        while offset<len(events):
            wd,mask,cookie,size=struct.unpack_from('iIII',events,offset)
            name=events[offset+16:offset+16+size].split(b'\0')[0]
            if os.fsdecode(name)==path.name:matched=True
            offset+=16+size
        if not matched:
            # Block on the next kernel event; unrelated writes never cause timed probes.
            continue
finally:os.close(fd)
