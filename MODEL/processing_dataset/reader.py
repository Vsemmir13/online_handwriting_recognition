import os
from decimal import Decimal

import numpy as np
from lxml import etree

from ..constants import SPLIT, PATH, DATA
from .base_classes import PointSet, Point
from ..decoding.decoding_converter import mlf2txt


class Sample(object):
    def __init__(self, xml_path, ground_truth):
        self.xml_path = xml_path
        self.ground_truth = ground_truth
        self.__pointset = None

    def generate_features(self, preprocess):
        return self.pointset.generate_features(preprocess=preprocess)

    def visualize(self):
        self.pointset.plot_points()

    def get_ground_truth_text(self):
        return mlf2txt(self.ground_truth)

    @property
    def name(self):
        return self.xml_path.split("/")[-1][:-4]

    @property
    def pointset(self):
        if self.__pointset is None:
            xml = open(self.xml_path, "rb").read()
            root = etree.XML(xml)
            wbd, strokeset = root.getchildren()
            sl = wbd[0].attrib["corner"]
            do = wbd[1].attrib["x"], wbd[1].attrib["y"]
            vo = wbd[2].attrib["x"], wbd[2].attrib["y"]
            ho = wbd[3].attrib["x"], wbd[3].attrib["y"]
            strokes = []
            stroke_id = 1
            min_time = Decimal(
                strokeset.getchildren()[0].getchildren()[0].attrib["time"]
            )
            for stroke in strokeset:
                for point in stroke:
                    t = (Decimal(point.attrib["time"]) - min_time) * 1000
                    x = point.attrib["x"]
                    y = point.attrib["y"]
                    strokes.append([stroke_id, t, x, y])
                stroke_id += 1
            strokes = np.asarray(strokes, dtype=int)
            r, b = do  # right, bottom edge
            l, _ = vo  # left edge
            _, u = ho  # upper edge
            r, b, l, u = int(r), int(b), int(l), int(u)
            strokes[:, 2] = np.subtract(strokes[:, 2], l)
            strokes[:, 3] = np.subtract(strokes[:, 3], u)
            points = []
            for s in strokes:
                points.append(Point(*s))
            self.__pointset = PointSet(
                points=points, w=r - l, h=b - u, file_name=self.xml_path
            )
        return self.__pointset

    def __repr__(self):
        return "<Sample of name={}>".format(self.name)


class IAMReader(object):
    def __init__(self, split, data_path=PATH.DATA_DIR):
        self.data_path = data_path
        self.line_data_path = data_path + "lineStrokes/"
        self.split = split
        self.samples = None

    def get_samples(self):
        if self.samples is not None:
            return self.samples
        sample_names = []
        if self.split == SPLIT.ALL:
            all_split = [SPLIT.TRAIN, SPLIT.VAL1, SPLIT.VAL2, SPLIT.TEST]
            for split in all_split:
                sample_names += self.__get_sample_names_from_split(split)
        else:
            sample_names = self.__get_sample_names_from_split(self.split)
        self.samples = self.__get_samples_from_name(sample_names)
        return self.samples

    def __get_sample_names_from_split(self, split):
        f = open(self.data_path + "split-config/" + split)
        return [line.strip(" \n") for line in f]

    def __get_samples_from_name(self, names, blacklist=DATA.BLACKLIST):
        f = open(self.line_data_path + "../t2_labels.mlf")
        count_missing_files = 0
        samples = []
        curr_path = ""
        curr_sample_name = ""
        curr_gt = []
        for line in f:
            if line[0] == "#":
                continue
            elif line[0] == '"':
                if curr_path and curr_sample_name in names:
                    curr_gt = curr_gt[:-1]
                    samples.append(Sample(curr_path, curr_gt))
                curr_path = ""
                curr_sample_name = ""
                curr_gt = []
                striped_line = line.strip(' "\n')
                line_split = striped_line.split("/")
                file_name = line_split[8].split(".")[0]
                if file_name in blacklist:
                    continue
                fn_split = file_name.split("-")
                path = (
                    fn_split[0]
                    + "/"
                    + fn_split[0]
                    + "-"
                    + fn_split[1][:3]
                    + "/"
                    + file_name
                    + ".xml"
                )
                path = self.line_data_path + path
                try:
                    if not os.path.getsize(path):
                        continue
                except FileNotFoundError:
                    # print("Missing file: {}".format(path))
                    count_missing_files += 1
                    continue
                curr_path = path
                curr_sample_name = fn_split[0] + "-" + fn_split[1]
            else:
                line_split = line.strip("\n")
                curr_gt.append(line_split)
        print("Count missing files:", count_missing_files)
        print("Ratio missing files:", count_missing_files / len(samples))
        return samples

    def __repr__(self):
        return '<IAMReader of split="{}">'.format(self.split)


def xmlpath2npypath(path, npz_dir):
    f_split = path.split("/")
    f_split[-4] = npz_dir
    f_split[-1] = f_split[-1][:-3] + "npz"
    f = "/".join(f_split)
    return f
