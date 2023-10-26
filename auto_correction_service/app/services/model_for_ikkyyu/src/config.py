class Config(object):
    LENGTH_MEASURE = {'千米': 1000, 'km': 1000, '分米': 0.1, 'dm': 0.1, '厘米': 0.01, 'cm': 0.01, '毫米':0.001, 'mm':0.001, '米': 1, 'm': 1}
    AERA_MEASURE = {'平方千米': 10 ** 6, '公顷': 10 ** 4, '亩': 666.666667, '平方米': 1, '平方分米': 0.01, '平方厘米': 1e-4, '平方毫米': 1e-6}
    VOLUME_MEASURE = {'立方米': 10 ** 3, '立方分米': 1, '立方厘米': 1e-3, '毫升': 1e-3, '升': 1}
    WEIGHT_MEASURE = {'吨': 10 ** 3, '千克': 1, 'kg': 1, '克': 1e-3, 'g': 1e-3, '公斤': 1, '斤': 0.5}
    MONEY_MEASURE = {'元': 100, '角': 10, '分': 1}
    TIME_MEASURE = {'世纪': 24 * 3600 * 365 * 100, '年': 24*3600 *365, '月': 24 * 3600 * 30, '天': 24 * 3600, '日': 24 * 3600, '时': 3600, '分': 60, '秒': 1}
    UNIT_CONVER = {}
    UNIT_CONVER.update(LENGTH_MEASURE)
    UNIT_CONVER.update(AERA_MEASURE)
    UNIT_CONVER.update(VOLUME_MEASURE)
    UNIT_CONVER.update(WEIGHT_MEASURE)
    UNIT_CONVER.update(MONEY_MEASURE)
    UNIT_CONVER.update(TIME_MEASURE)
    UNIT_CONVER_LIST = []
    UNIT_CONVER_LIST.append(LENGTH_MEASURE)
    UNIT_CONVER_LIST.append(AERA_MEASURE)
    UNIT_CONVER_LIST.append(VOLUME_MEASURE)
    UNIT_CONVER_LIST.append(WEIGHT_MEASURE)
    UNIT_CONVER_LIST.append(MONEY_MEASURE)
    UNIT_CONVER_LIST.append(TIME_MEASURE)
    DECODE = '10853-2=6×7÷49+()*@~○><{}|x.千米分平方公毫立升吨克元角日时秒厘 '
    UNIT = '元角分吨千克公斤厘米分毫里时秒平方立顷亩升年月日天kgcmdm世纪'