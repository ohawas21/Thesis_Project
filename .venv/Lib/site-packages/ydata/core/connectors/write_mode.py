from ydata.core.enum import StringEnum


class WriteMode(StringEnum):
    FAIL = "fail"
    REPLACE = "replace"
    APPEND = "append"
