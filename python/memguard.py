#!/usr/bin/env python3
"""Memory guard for the long export, quantization and repair jobs (stdlib, macOS and Linux).

    from memguard import guard; guard(min_free_gb=8)     # first line of a job's main()
    uv run python python/memguard.py                      # print the machine's available memory and exit

`guard` refuses to start a job when the machine has less than `min_free_gb` available. Then a daemon thread
samples the available memory every few seconds, logs a WARN line when it falls below twice the floor, and stops the
job when it falls below the floor: first with KeyboardInterrupt in the main thread (so `finally` blocks run and
checkpoints stay), after 30 seconds with exit code 3. At exit it logs the job's peak resident memory and the lowest
available memory it saw. The second is the one to plan with on Apple silicon. VERIFIED: a repair run on MPS took
8.4 GB of available memory at a peak resident size of 6.3 GB (2026-11-09).

WHY a floor on the machine's available memory and not a limit on the job's own size: the damage comes from several
jobs (and browsers, and the gate's e2e run) together pushing the machine into swap; only the machine-wide number
sees that. VERIFIED: on 2026-11-09 the maintainer had to stop two repair runs and a vision export that ran at once on a
48 GB Mac because together they took too much memory.
"""
import atexit, os, platform, re, resource, signal, subprocess, sys, threading, time

GB = 1024 ** 3


def log(level, msg):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + f' {level} {msg}', flush=True)


def available_bytes():
    """Memory the machine can hand out without swapping: MemAvailable on Linux; free, inactive and speculative pages
    on macOS (vm_stat)."""
    if platform.system() == 'Linux':
        with open('/proc/meminfo') as f:
            return int(re.search(r'^MemAvailable:\s+(\d+) kB', f.read(), re.M).group(1)) * 1024
    out = subprocess.run(['vm_stat'], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r'page size of (\d+) bytes', out).group(1))
    pages = {k: int(v) for k, v in re.findall(r'^Pages (free|inactive|speculative):\s+(\d+)\.', out, re.M)}
    return (pages.get('free', 0) + pages.get('inactive', 0) + pages.get('speculative', 0)) * page


def peak_rss_bytes():
    """This process's peak resident memory (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if platform.system() == 'Darwin' else r * 1024


def guard(min_free_gb=8.0, every=5.0):
    """Refuse to start below `min_free_gb` available, then watch it until the process exits (see the module doc)."""
    free = available_bytes() / GB
    if free < min_free_gb:
        log('ERROR', f'memory refused start available_gb={free:.1f} min_free_gb={min_free_gb} hint="stop other jobs or lower --min-free-gb"')
        sys.exit(3)
    log('INFO', f'memory guard available_gb={free:.1f} min_free_gb={min_free_gb}')
    low = [free]  # the lowest available memory seen; with the start value, what the job took from the machine
    atexit.register(lambda: log('INFO', f'memory peak_rss_gb={peak_rss_bytes() / GB:.1f} min_available_gb={low[0]:.1f}'))

    def watch():
        warned = False
        while True:
            time.sleep(every)
            free = available_bytes() / GB; low[0] = min(low[0], free)
            if free < min_free_gb:
                log('ERROR', f'memory below floor available_gb={free:.1f} min_free_gb={min_free_gb} rss_peak_gb={peak_rss_bytes() / GB:.1f}; stopping')
                os.kill(os.getpid(), signal.SIGINT); time.sleep(30); os._exit(3)  # SIGINT also wakes a blocking call
            if free < 2 * min_free_gb and not warned:
                log('WARN', f'memory low available_gb={free:.1f} min_free_gb={min_free_gb}'); warned = True
            elif free >= 2 * min_free_gb: warned = False
    threading.Thread(target=watch, daemon=True, name='memguard').start()


if __name__ == '__main__':
    log('INFO', f'memory available_gb={available_bytes() / GB:.1f}')
