"""This computer's memory: how much is free, and holding a process to a limit, so a job that needs more than there is fails
(MemoryError) instead of exhausting the machine. On Linux and Windows."""
import ctypes
import os
from pathlib import Path

RESERVE_GB = 4  # left free for everything else while a job runs
_held = []  # Windows: the job object a process is held by, kept open with it


def available_gb():
    """Memory free for new work, GB."""
    if os.name == "nt":
        status = _Status(dwLength=ctypes.sizeof(_Status))
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise ctypes.WinError()
        return status.ullAvailPhys / 2 ** 30
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2 ** 20
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def limit(gb):
    """Hold this process to gb of memory: past it, asking for more fails."""
    if os.name != "nt":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (int(gb * 2 ** 30), resource.getrlimit(resource.RLIMIT_AS)[1]))
        return
    from ctypes import wintypes
    k = ctypes.windll.kernel32
    k.CreateJobObjectW.restype = k.GetCurrentProcess.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k.CreateJobObjectW(None, None)
    info = _JobLimits()
    info.basic.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(gb * 2 ** 30)
    if not job or not k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) or not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
        raise ctypes.WinError()  # 9: JobObjectExtendedLimitInformation
    _held.append(job)


class _Status(ctypes.Structure):  # MEMORYSTATUSEX
    _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), *((f, ctypes.c_uint64) for f in (
        "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual"))]


class _BasicLimits(ctypes.Structure):  # JOBOBJECT_BASIC_LIMIT_INFORMATION
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32)]


class _JobLimits(ctypes.Structure):  # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    _fields_ = [("basic", _BasicLimits), ("IoInfo", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
