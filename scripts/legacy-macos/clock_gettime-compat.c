/* clock_gettime() for macOS 10.11.
 *
 * clock_gettime() appeared in macOS 10.12. Built with a recent SDK and a
 * minimum version of 10.11 it is left as a "weak import": on 10.11 it is NULL
 * and the program crashes as soon as std::chrono asks for the time.
 * This object is linked into each binary of the 10.11 build (libc++,
 * libc++abi, CTranslate2): its own clock_gettime, hidden, made with
 * gettimeofday/mach_absolute_time, which exist in every version. */
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

/* another C name (the SDK declares clock_gettime as 10.12+), the same symbol */
__attribute__((visibility("hidden")))
int uw_clock_gettime(int clk, struct timespec *ts) __asm__("_clock_gettime");

static void set_ns(struct timespec *ts, uint64_t ns) {
    ts->tv_sec = (time_t) (ns / 1000000000ULL);
    ts->tv_nsec = (long) (ns % 1000000000ULL);
}

int uw_clock_gettime(int clk, struct timespec *ts) {
    switch (clk) {
    case 0: { /* CLOCK_REALTIME */
        struct timeval tv;
        if (gettimeofday(&tv, 0) != 0) return -1;
        ts->tv_sec = tv.tv_sec;
        ts->tv_nsec = (long) tv.tv_usec * 1000;
        return 0;
    }
    case 12: { /* CLOCK_PROCESS_CPUTIME_ID */
        struct rusage ru;
        if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
        set_ns(ts, ((uint64_t) ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000000ULL +
                   ((uint64_t) ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) * 1000ULL);
        return 0;
    }
    case 16: { /* CLOCK_THREAD_CPUTIME_ID */
        thread_basic_info_data_t info;
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        mach_port_t thread = mach_thread_self();
        kern_return_t kr = thread_info(thread, THREAD_BASIC_INFO, (thread_info_t) &info, &count);
        mach_port_deallocate(mach_task_self(), thread);
        if (kr != KERN_SUCCESS) return -1;
        set_ns(ts, ((uint64_t) info.user_time.seconds + info.system_time.seconds) * 1000000000ULL +
                   ((uint64_t) info.user_time.microseconds + info.system_time.microseconds) * 1000ULL);
        return 0;
    }
    default: { /* CLOCK_MONOTONIC, CLOCK_MONOTONIC_RAW, CLOCK_UPTIME_RAW... */
        static mach_timebase_info_data_t tb;
        if (tb.denom == 0) mach_timebase_info(&tb);
        set_ns(ts, (uint64_t) ((__uint128_t) mach_absolute_time() * tb.numer / tb.denom));
        return 0;
    }
    }
}
