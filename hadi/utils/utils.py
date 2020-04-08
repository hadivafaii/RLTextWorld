def convert_time(time_in_secs):

    d = time_in_secs // 86400
    h = (time_in_secs - d * 86400) // 3600
    m = (time_in_secs - d * 86400 - h * 3600) // 60
    s = time_in_secs - d * 86400 - h * 3600 - m * 60

    print("\nd / hh:mm:ss   --->   %d / %d:%d:%d\n" % (d, h, m, s))
