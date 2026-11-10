#!/usr/bin/env python3
"""Execute a local validation command with kernel-enforced socket denial.
Linux/libseccomp only. Inherited by descendants; cannot be undone by the child.
Use AFTER downloading dependencies and pinned model snapshots.
"""
import ctypes, ctypes.util, errno, os, socket, sys
if sys.platform != 'linux':raise SystemExit('This offline validation wrapper requires Linux/libseccomp')
if len(sys.argv)<2:raise SystemExit('Usage: python offline-linux.py COMMAND [ARG ...]')
lib=ctypes.CDLL(ctypes.util.find_library('seccomp'),use_errno=True)
lib.seccomp_init.argtypes=[ctypes.c_uint32];lib.seccomp_init.restype=ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
lib.seccomp_load.argtypes=[ctypes.c_void_p];lib.seccomp_release.argtypes=[ctypes.c_void_p]
ctx=lib.seccomp_init(0x7fff0000)
if not ctx:raise RuntimeError('seccomp_init failed')
for name in ['socket','connect','sendto','sendmsg','sendmmsg']:
    number=lib.seccomp_syscall_resolve_name(name.encode())
    if number>=0 and lib.seccomp_rule_add(ctx,0x00050000|errno.EPERM,number,0):raise RuntimeError('seccomp_rule_add failed')
if lib.seccomp_load(ctx):raise RuntimeError('seccomp_load failed')
lib.seccomp_release(ctx)
try:socket.socket()
except PermissionError:pass
else:raise RuntimeError('Socket denial verification failed')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
print('Verified: kernel denies socket creation and network sends for this command.',flush=True)
os.execvp(sys.argv[1],sys.argv[1:])
