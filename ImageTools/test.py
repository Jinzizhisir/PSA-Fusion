# 计算两个日期之间相差的天数
# 日期格式为 YYYY-MM-DD

import datetime

def days_diff(date1, date2):
    date1 = datetime.date(int(date1[0]), int(date1[1]), int(date1[2]))
    date2 = datetime.date(int(date2[0]), int(date2[1]), int(date2[2]))
    return abs((date1 - date2).days)   

print(days_diff(('1981', '4', '19'), ('1982', '4', '22')))


