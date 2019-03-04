#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 24/2/2019 3:28 PM
import sys
from transformer.data_load import load_vocab

if sys.version_info[0] == 3:
    unichr = chr


class Encoder(object):
    def __init__(self, extra_num_chars, ignore_comma=False, use_default_nlp_symbol=True):
        self.extra_num_chars = extra_num_chars
        self.ignore_comma = ignore_comma
        self.use_default_nlp_symbol = use_default_nlp_symbol  # 0: <pad>, 1: <unk>, 2: <s>, 3: </s>
        self.special_symbols = ["<pad>", "<unk>", "<s>", "</s>"]
        self.comma_char = 0x2C
        self.exception_char = {0x2D: 0,  # "-"
                               }
        self.num_char = [0x30, 0x39]
        self.upeng_char = [0x41, 0x5a]
        self.loweng_char = [0x61, 0x7a]
        self.chn_char = [0x4e00, 0x9fa5]

    def get_vocab_size(self):
        """
        Including special symbols
        :return:
        """
        ret_num = len(self.exception_char) \
                  + (self.num_char[1] - self.num_char[0] + 1) \
                  + (self.upeng_char[1] - self.upeng_char[0] + 1) \
                  + (self.loweng_char[1] - self.loweng_char[0] + 1) \
                  + (self.chn_char[1] - self.chn_char[0] + 1)
        if not self.ignore_comma:
            ret_num += 1
        if self.use_default_nlp_symbol:
            ret_num += len(self.special_symbols)
        return ret_num

    def get_index_by_char(self, unicode_char):
        char = unicode_char
        if sys.version_info[0] < 3:
            if not isinstance(char, unicode):
                char = char.decode("utf-8", "ignore")

        return_idx = self.extra_num_chars
        if self.use_default_nlp_symbol:
            for i, symbol in enumerate(self.special_symbols):
                if char == symbol:
                    return i
            return_idx += len(self.special_symbols)
        unicode_code = ord(char)
        if not self.ignore_comma:
            if self.comma_char == unicode_code:
                return return_idx
            else:
                return_idx += 1
        if unicode_code in self.exception_char:
            return return_idx + self.exception_char[unicode_code]
        else:
            return_idx += len(self.exception_char)

        if self.num_char[0] <= unicode_code <= self.num_char[1]:
            return return_idx + (unicode_code - self.num_char[0])
        else:
            return_idx += self.num_char[1] - self.num_char[0] + 1

        if self.upeng_char[0] <= unicode_code <= self.upeng_char[1]:
            return return_idx + (unicode_code - self.upeng_char[0])
        else:
            return_idx += self.upeng_char[1] - self.upeng_char[0] + 1

        if self.loweng_char[0] <= unicode_code <= self.loweng_char[1]:
            return return_idx + (unicode_code - self.loweng_char[0])
        else:
            return_idx += self.loweng_char[1] - self.loweng_char[0] + 1

        if self.chn_char[0] <= unicode_code <= self.chn_char[1]:
            return return_idx + (unicode_code - self.chn_char[0])
        else:
            return_idx += self.chn_char[1] - self.chn_char[0] + 1

        return -1

    def get_char_by_index(self, idx):
        offset = idx
        if self.use_default_nlp_symbol:
            if offset < len(self.special_symbols):
                return self.special_symbols[offset]
            offset -= len(self.special_symbols)
        if offset < self.extra_num_chars:
            return None
        offset -= self.extra_num_chars

        if not self.ignore_comma:
            if 0 == offset:
                return u","
            offset -= 1

        if offset < len(self.exception_char):
            for k, v in self.exception_char.items():
                if v == offset:
                    return unichr(k)
        offset -= len(self.exception_char)

        num_num_char = self.num_char[1] - self.num_char[0] + 1
        if offset < num_num_char:
            return unichr(self.num_char[0] + offset)
        offset -= num_num_char

        num_upeng_char = self.upeng_char[1] - self.upeng_char[0] + 1
        if offset < num_upeng_char:
            return unichr(self.upeng_char[0] + offset)
        offset -= num_upeng_char

        num_loweng_char = self.loweng_char[1] - self.loweng_char[0] + 1
        if offset < num_loweng_char:
            return unichr(self.loweng_char[0] + offset)
        offset -= num_loweng_char

        num_chn_char = self.chn_char[1] - self.chn_char[0] + 1
        if offset < num_chn_char:
            return unichr(self.chn_char[0] + offset)
        offset -= num_chn_char

        return None

    def get_filtered_unicode_str(self, unc_str):
        str1 = unc_str
        if sys.version_info[0] < 3:
            if not isinstance(str1, unicode):
                str1 = str1.decode("utf-8", "ignore")
        return_str = u""
        for char in str1:
            idx = self.get_index_by_char(char)
            if -1 != idx:
                return_str += char

        return return_str

    def get_index_dict(self):
        token2idx, idx2token = {}, {}
        vocab_size = self.get_vocab_size()
        for i in range(self.extra_num_chars, vocab_size):
            token = self.get_char_by_index(i)
            token2idx[token] = i
            idx2token[i] = token
        return token2idx, idx2token

    def output_dict(self, file_name="nlptools.vocab"):
        token2idx, idx2token = self.get_index_dict()
        with open(file_name, "w") as f:
            for token, idd in sorted(token2idx.items(), key=lambda _d: _d[1]):
                str_out = "%s\t%s\n" % (token, idd)
                if sys.version_info[0] == 2:
                    str_out = str_out.encode("utf-8")
                f.write(str_out)
            f.close()


_encoder = Encoder(extra_num_chars=0, ignore_comma=False, use_default_nlp_symbol=True)
get_num_vocab = _encoder.get_vocab_size
get_char_index = _encoder.get_index_by_char
get_filtered_unicode_str = _encoder.get_filtered_unicode_str
get_char_by_index = _encoder.get_char_by_index
get_index_dict = _encoder.get_index_dict


def main():
    # str1 = "Unicode 编码包含了不同写法的字，如“ɑ/a”、“户/户/戸”。"
    # str2 = get_filtered_unicode_str(str1)
    # print(str2)
    print(get_num_vocab())  # 20965
    idx = get_char_index(u'国')
    print(idx)
    print(ord(u"国") - 0x4e00)
    print(get_char_by_index(0))
    list_char = []
    for i in range(100):
        c = get_char_by_index(i)
        # if type(c) in (unicode, str):
        #     list_char.append(c)
        print("%s\t%s" % (i, c))
    for c in list_char:
        print("%s\t%s" % (get_char_index(c), c))
    print(get_char_index(','))
    d1, d2 = get_index_dict()
    db = "db"
    _encoder.output_dict()


if __name__ == '__main__':
    main()
