#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 22/3/2019 12:00 PM
import json
import sys

PY3 = sys.version_info[0] == 3
unicode_type = str

if not PY3:
    unicode_type = unicode


def CHECK_TASK_TYPE(task_type):
    assert task_type in ("set2set", "set2sca", "seq2seq", "seq2sca", "set2cls", "seq2cls")


class Instance(object):
    def __init__(self, task_type, input1, target, input2=None):
        """
        样本格式，适配于DickLearning
        :param task_type: 任务类型，详情见代码的列举，有三种格式：set, bag, seq, sca。其中set和bag的区别在于
        set在模型中的值只有0/1，而bag不限。2sca意思是输出回归值，其target是float，2cls是分类模型，其target是int
        :param input1: list, 元素必须为unicode
        :param target: scalar or list
        :param input2: list, 可为空
        """
        # task_type
        CHECK_TASK_TYPE(task_type)
        self.task_type = task_type
        # input1
        assert isinstance(input1, list)
        for ele in input1:
            if not isinstance(ele, unicode_type):
                raise Exception("元素必须为unicode")
        self.input1 = input1
        # target
        if task_type.endswith("2sca"):
            assert isinstance(target, float)
        elif task_type.endswith("2cls"):
            assert isinstance(target, int)
        elif task_type.endswith("2set"):
            assert isinstance(target, list)
        else:
            raise Exception("target must correspond to task_type")
        self.target = target
        # input2
        if input2 is not None:
            assert isinstance(input2, list)
            for ele in input2:
                if not isinstance(ele, unicode_type):
                    raise Exception("元素必须为unicode")
        self.input2 = input2

    @classmethod
    def load_instance_from_json(cls, str_json):
        data = json.loads(str_json)
        instance = cls(task_type=data["task_type"],
                       input1=data["input1"],
                       target=data["target"],
                       input2=data.get("input2"))
        return instance

    def to_json_string(self):
        data = {
            "task_type": self.task_type,
            "input1": self.input1,
            "target": self.target,
        }
        if self.input2:
            data["input2"] = self.input2

        if PY3:
            return json.dumps(data, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False).encode("utf-8")


def main():
    inst = Instance(task_type="set2sca", input1=[u"太阳", u"月亮"], input2=[u"人类", u"地球"], target=0.7)
    print(inst.to_json_string())


def main2():
    import sys
    for line in sys.stdin:
        inst = Instance.load_instance_from_json(line)
        print(inst.to_json_string())


if __name__ == '__main__':
    main2()
