# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pandas as pd
import xml.etree.ElementTree as ET
from best.types import TwoWayDict

class NSRRSleepFile:
    def __init__(self, path=None, nsrr2hypnogram_keys={
        'Wake|0': 'WAKE',
        'Stage 1 sleep|1': 'N1',
        'Stage 2 sleep|2': 'N2',
        'Stage 3 sleep|3': 'N3',
        'Stage 4 sleep|4': 'N3',
        'REM sleep|5': 'REM'
    }):
        self.namespaces = TwoWayDict() # list of namespaces - key=prefix; value=uri
        self.Element = None
        self.path = path
        self.strp_format = '%Y.%m.%d. %H:%M:%S'

        self.nsrr2hypnogram = TwoWayDict(nsrr2hypnogram_keys)



        self.epoch_length = None
        self.software_version = None
        self.events = None

        if path:
            if os.path.isfile(path):
                self.read_file(path_xml=path)

    def read_file(self, path_xml):
        self.Element = ET.parse(path_xml).getroot()
        for Element in self.Element:
            tag = Element.tag
            if tag == 'SoftwareVersion':
                self.software_version = Element.text.split('\n')[0]
            if tag == 'EpochLength':
                self.epoch_length = float(Element.text.split('\n')[0])

        self._parse_file()

    def _parse_file(self):
        events = []
        for Element in self.Element:
            tag = Element.tag
            if tag == 'ScoredEvents':
                for ScoredEvent in Element:
                    tag_ScoredEvent = ScoredEvent.tag
                    event = {}
                    for Property in ScoredEvent:
                        tag_Property = Property.tag
                        if Property.text:
                            text_Property = Property.text.split('\n')[0]
                            event[tag_Property] = text_Property
                    events += [event]
        self.events = events

    def get_hypnogram(self):
        hyp = pd.DataFrame([ev for ev in self.events if 'EventType' in ev.keys() if ev['EventType'] == 'Stages|Stages'])
        hyp['duration'] = [float(x) for x in hyp['Duration']]
        hyp = hyp.drop(['Duration'], axis=1)

        hyp['start'] = [float(x) for x in hyp['Start']]
        hyp = hyp.drop(['Start'], axis=1)

        hyp = hyp.drop(['EventType'], axis=1)

        hyp['annotation'] = [self.nsrr2hypnogram[an] for an in hyp['EventConcept']]
        hyp = hyp.drop(['EventConcept'], axis=1)

        hyp['end'] = hyp['start'] + hyp['duration']
        hyp = hyp[['annotation', 'start', 'end', 'duration']]
        return hyp

    def get_annotation_types(self):
        types = {}
        for Element in self.Element:
            tag = Element.tag
            if '}' in tag: tag = tag.split('}')[-1]
            if tag == 'AnnotationTypes':
                for AnnotationTypeElement in Element:
                    name = None
                    id = None
                    for SubElement in AnnotationTypeElement:
                        subtag = SubElement.tag
                        if '}' in subtag: subtag = subtag.split('}')[-1]
                        if subtag == 'id': id = SubElement.text
                        if subtag == 'name': name = SubElement.text
                    types[id] = name
        return types
