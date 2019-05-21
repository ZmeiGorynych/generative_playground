def pulsing_schedule(x, period=300, max=10):
    x = (x % period) - period
    x = (x*x) / (period*period) + 1
    return max*x
