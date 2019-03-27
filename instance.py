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
    assert task_type in ("set2vec", "set2sca", "seq2seq", "seq2sca", "set2cls", "seq2cls", "seq2vec", "seq2vecseq")


def CHECK_2SEQ(target):
    assert target[-1] == "</s>"


class Instance(object):
    def __init__(self, task_type, inputs, target):
        """
        样本格式，适配于DickLearning
        :param task_type: 任务类型，详情见代码的列举，有五种格式：set, bag, seq, sca，vec。其中set和bag的区别在于
        set在模型中的值只有0/1，而bag不限，set在输入时会强制去重。2sca意思是输出回归值，其target是float，2cls是分类模型，其target是int，
        vec输出是一个浮点vector
        :param inputs: [input1, input2, ...], 每个input必须为list, 元素必须为unicode
        :param target: scalar or list
        """
        # task_type
        CHECK_TASK_TYPE(task_type)
        self.task_type = task_type
        # inputs
        for inputt in inputs:
            assert isinstance(inputt, list)
            for ele in inputt:
                if not isinstance(ele, unicode_type):
                    raise Exception("元素必须为unicode")
        self.inputs = inputs
        # target, TODO: 2seq
        if task_type.endswith("2sca"):
            assert isinstance(target, float)
        elif task_type.endswith("2cls"):
            assert isinstance(target, int)
        elif task_type.endswith("2vec"):
            assert isinstance(target, list)
            for ele in target:
                assert isinstance(ele, float)
        elif task_type.endswith("2vecseq"):
            assert isinstance(target, list)
            for vec in target:
                assert isinstance(vec, list)
                for ele in vec:
                    assert isinstance(ele, float)
        else:
            raise Exception("target must correspond to task_type")
        self.target = target

    @classmethod
    def load_instance_from_json(cls, str_json):
        data = json.loads(str_json)
        instance = cls(task_type=data["task_type"],
                       inputs=data["inputs"],
                       target=data["target"])
        return instance

    def to_json_string(self, check_2seq=True):
        if self.task_type.endswith("2seq"):
            if check_2seq:
                CHECK_2SEQ(self.target)

        data = {
            "task_type": self.task_type,
            "inputs": self.inputs,
            "target": self.target,
        }

        if PY3:
            return json.dumps(data, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

def main():
    inst = Instance(task_type="set2sca", inputs=[[u"太阳", u"月亮"], [u"人类", u"地球"]], target=0.7)
    print(inst.to_json_string())


def main2():
    import sys
    for line in sys.stdin:
        inst = Instance.load_instance_from_json(line)
        print(inst.to_json_string())


if __name__ == '__main__':
    main2()
