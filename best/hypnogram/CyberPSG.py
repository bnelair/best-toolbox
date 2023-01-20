# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from io import StringIO
import xml.etree.ElementTree as ET
import os
import pandas as pd
from dateutil import tz
from datetime import datetime
import os
import re
from datetime import datetime
import pytz
from copy import deepcopy
import argparse
import xml.etree.ElementTree as ET
from _annotation_tools.XML_CyberPSG import *
import uuid

xsi =  "http://www.w3.org/2001/XMLSchema-instance"
xmlns =  "http://tempuri.org/CyberPSG.xsd"
ns = {"xmlns:xsi": xsi}

class TwoWayDict(dict):
    def __init__(self, d=None):
        if not isinstance(d, type(None)):
            if isinstance(d, dict):
                for k, v in d.items():
                    self[k] = v



    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        return dict.__len__(self) // 2

class CyberPSGFile:
    def __init__(self, path=None):
        self.namespaces = TwoWayDict() # list of namespaces - key=prefix; value=uri
        self.Element = None
        self.path = path
        self.strp_format = '%Y-%m-%dT%H:%M:%S.%f'

        if path:
            if os.path.isfile(path):
                self.read_file(path_xml=path)


    def register_namespace(self, prefix, uri):
        if not prefix in self.namespaces.keys():
            self.namespaces[prefix] = uri
            ET.register_namespace(prefix, uri)

    def parse_namespaces(self, path_xml):
        schema = open(path_xml, 'r').read()
        namespaces = dict([
            node for _, node in ET.iterparse(StringIO(schema), events=['start-ns'])
        ])
        for k, v in namespaces.items(): self.register_namespace(k, v)

    def register_CyberPSG_namespace(self):
        self.register_namespace('', "http://tempuri.org/CyberPSG.xsd")
        self.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")


    def read_file(self, path_xml):
        self.parse_namespaces(path_xml)
        self.Element = ET.parse(path_xml).getroot()

    def get_annotation_types(self):
        types = {}
        for Element in self.Element:
            tag = Element.tag
            if '}' in tag: tag = tag.split('}')[-1]
            if tag == 'AnnotationTypes':
                for AnnotationTypeElement in Element:
                    name = None
                    id = None
                    electrode = None
                    for SubElement in AnnotationTypeElement:
                        subtag = SubElement.tag
                        if '}' in subtag: subtag = subtag.split('}')[-1]
                        if subtag == 'id': id = SubElement.text
                        if subtag == 'name': name = SubElement.text
                    types[id] = name
        return types


    def get_annotations(self):
        types = self.get_annotation_types()
        types = TwoWayDict(types)
        annotations = []

        for Element in self.Element:
            tag = Element.tag
            if '}' in tag: tag = tag.split('}')[-1]
            if tag == 'Annotations':
                for AnnotationElement in Element:
                    annot = {
                        'annotation': None,
                        'startTimeUtc': None,
                        'endTimeUtc': None
                    }

                    for SubElement in AnnotationElement:
                        subtag = SubElement.tag
                        if '}' in subtag: subtag = subtag.split('}')[-1]
                        if subtag == 'annotationTypeId':  annot['annotationTypeId'] = types[types[SubElement.text]]
                        if subtag == 'startTimeUtc':  annot['startTimeUtc'] = SubElement.text
                        if subtag == 'endTimeUtc':  annot['endTimeUtc'] = SubElement.text
                        if subtag == 'channelName': annot['channel'] = SubElement.text
                    annotations += [annot]
        return annotations

    def get_hypnogram(self):
        types = self.get_annotation_types()
        types = TwoWayDict(types)
        annotations = []

        for Element in self.Element:
            tag = Element.tag
            if '}' in tag: tag = tag.split('}')[-1]
            if tag == 'Annotations':
                for AnnotationElement in Element:
                    annot = {
                        'annotation': None,
                        'start': None,
                        'end': None
                    }

                    for SubElement in AnnotationElement:
                        subtag = SubElement.tag
                        if '}' in subtag: subtag = subtag.split('}')[-1]
                        if subtag == 'annotationTypeId':  annot['annotation'] = types[SubElement.text]
                        if subtag == 'startTimeUtc':
                            s = SubElement.text
                            if '.' in s:
                                frmt = self.strp_format
                            else:
                                frmt = self.strp_format.split('.')[0]
                            utc = datetime.strptime(s, frmt)
                            utc = utc.replace(tzinfo=tz.tzutc())
                            annot['start'] = utc
                        if subtag == 'endTimeUtc':
                            s = SubElement.text
                            if '.' in s:
                                frmt = self.strp_format
                            else:
                                frmt = self.strp_format.split('.')[0]
                            utc = datetime.strptime(s, frmt)
                            utc = utc.replace(tzinfo=tz.tzutc())
                            annot['end'] = utc
                        if subtag == 'channelName': annot['channel'] = SubElement.text
                    annotations += [annot]
        return annotations

class CyberPSG_XML_Writter:
    __version__ = '1.0.1'
    """
    Version 1.0.0 update
    Database of UUIDs implemented
    
    Version 1.01 update
    adding channel name to the export
    """


    _xsi =  "http://www.w3.org/2001/XMLSchema-instance"
    _xmlns =  "http://tempuri.org/CyberPSG.xsd"
    _ns = {"xmlns:xsi": xsi}

    standard_UUID = {
        'AWAKE_best': '00000000-898a-4b80-8ed8-000000000000',
        'N1_best' : '00000000-898a-4b80-8ed8-111111111111',
        'N2_best': '00000000-898a-4b80-8ed8-222222222222',
        'N3_best': '00000000-898a-4b80-8ed8-333333333333',
        'REM_best': '00000000-898a-4b80-8ed8-444444444444',
        'UNKNOWN_best': '00000000-898a-4b80-8ed8-555555555555',
        'Arousal_best': '00000000-898a-4b80-8ed8-666666666666',
        'N_best': '00000000-898a-4b80-8ed8-777777777777',
        'SLP_best': '00000000-898a-4b80-8ed8-888888888888',
        'Sleep stage N1': '00000001-898a-4b80-8ed8-000000000001',
        'Sleep stage N2': '00000001-898a-4b80-8ed8-000000000002',
        'Sleep stage N3': '00000001-898a-4b80-8ed8-000000000003',
        'Sleep stage R':  '00000001-898a-4b80-8ed8-000000000004',
        'Sleep stage W':  '00000001-898a-4b80-8ed8-000000000005',
        'IED':  '00000001-898a-4b80-8ed8-000000000010',
        'IED_best':  '00000001-898a-4b80-8ed8-000000000011',
    }




    annotation_keys = [k[:-5] for k in standard_UUID.keys()]


    def __init__(self, path):
        for attr, uri in self._ns.items():
            ET.register_namespace(attr.split(":")[1], uri)

        self.AnnotationData = myXML_AnnotationData.init_EmptyInstance(file_path=path)
        self.AnnotationData.attributes['xmlns'] = self._xmlns

        self.AnnotationGroups = self.AnnotationData.add_child(myXML_AnnotationGroups)
        self.AnnotationTypes = self.AnnotationData.add_child(myXML_AnnotationTypes)
        self.Annotations = self.AnnotationData.add_child(myXML_Annotations)

    @property
    def nAnnotationGroups(self):
        return self.AnnotationGroupKeys.__len__()

    @property
    def AnnotationGroupKeys(self):
        return [AnnotationGroup['name'][0].param for AnnotationGroup in self.AnnotationGroups['AnnotationGroup']]

    @property
    def AnnotationGroupIDs(self):
        return [AnnotationGroup['id'][0].param for AnnotationGroup in self.AnnotationGroups['AnnotationGroup']]

    def add_AnnotationGroup(self, name_string='', uuid_=None):
        if name_string.__len__() == 0:
            name_string = 'AnnotationGroup{0}'.format(self.AnnotationGroups['AnnotationGroup'].__len__())

        for AnnotationGroup in self.AnnotationGroups['AnnotationGroup']:
            if AnnotationGroup['name'] == name_string:
                raise KeyError("AnnotationGroup \"" + name_string + "\" already exists!")

        AnnotationGroup = self.AnnotationGroups.add_child(myXML_AnnotationGroup)
        ID = AnnotationGroup.add_child(myXML_Id)
        if not uuid_:
            ID.param = str(uuid.uuid4())
        else:
            ID.param = str(uuid_)
        name = AnnotationGroup.add_child(myXML_Name)
        name.param = name_string

    def remove_AnnotationGroup(self, item):
        if not (isinstance(item, int) and item >= 0 and item < self.AnnotationGroups['AnnotationGroup'].__len__()) and not isinstance(item, str):
            raise KeyError('Annotation group identifier must be an integer indexing Annotation groups or string representing a category name.')

        for idx, AnnotationGroup in enumerate(self.AnnotationGroups['AnnotationGroup']):
            if AnnotationGroup['name'][0].param == item:
                item = idx

        self.AnnotationGroups['AnnotationGroup'][item].Element.clear()
        del self.AnnotationGroups.AnnotationGroup[item]

    @property
    def nAnnotationTypes(self):
        return self.AnnotationTypeKeys.__len__()

    @property
    def AnnotationTypeKeys(self):
        return [AnnotationType['name'][0].param for AnnotationType in self.AnnotationTypes.AnnotationType]

    @property
    def AnnotationTypeIDs(self):
        return [AnnotationType['id'][0].param for AnnotationType in self.AnnotationTypes.AnnotationType]

    def add_AnnotationType(self, name_string="", groupAssociationId=None, color='#FF6C1FBA', note="", startsWithEpoch=False, stdDurationInSec=0):
        if name_string.__len__() == 0:
            name_string = 'AnnotationType{0}'.format(self.AnnotationTypes.__len__())

        # if name_string+'_aisc' in list(self.standard_UUID.keys()):
        #     name_string += '_aisc'
        if name_string+'_best' in list(self.standard_UUID.keys()):
            name_string += '_best'

        #print(name_string)
        #print(list(self.standard_UUID.keys()))
        #print(name_string+'_aisc' in list(self.standard_UUID.keys()))

        for AnnotationType in self.AnnotationTypes.AnnotationType:
            if AnnotationType.name[0].param == name_string:
                raise KeyError("AnnotationType \"" + name_string + "\" already exists!")

        AnnotationType = self.AnnotationTypes.add_child(myXML_AnnotationType)

        if name_string in list(self.standard_UUID.keys()):
            uuid_key = self.standard_UUID[name_string]
        else:
            uuid_key = str(uuid.uuid4())

        #name_string = name_string + '_aisc'
        ID = AnnotationType.add_child(myXML_Id, param=uuid_key)
        name = AnnotationType.add_child(myXML_Name, param=name_string)
        color = AnnotationType.add_child(myXML_Color, param=color)
        if note.__len__() > 0:
            note = AnnotationType.add_child(myXML_Note, param=note)

        if isinstance(groupAssociationId, str):
            is_AGroup_set = False
            for idx, AGroupName in enumerate(self.AnnotationGroupKeys):
                if AGroupName == groupAssociationId:
                    groupAssociationId = idx
                    is_AGroup_set = True

            if is_AGroup_set is False:
                raise KeyError('AnnotationType cannot be assigned to an AnnotationGroup \"' + groupAssociationId + '\". Available AnnotationGroups \"' + str(self.AnnotationGroupKeys ))

        if isinstance(groupAssociationId, int):
            group_id = self.AnnotationGroups.AnnotationGroup[groupAssociationId].id[0].param
            groupAssociation = AnnotationType.add_child(myXML_GroupAsociations)
            gAID = groupAssociation.add_child(myXML_Id, param=group_id)

    def remove_AnnotationType(self, item):
        if not (isinstance(item, int) and item >= 0 and item < self.AnnotationTypes.AnnotationType.__len__()) and not isinstance(item, str):
            raise KeyError('Annotation type identifier must be an integer indexing Annotation types or string representing a category name.')

        for idx, AnnotationTypeKey in enumerate(self.AnnotationTypesKeys):
            if AnnotationTypeKey == item:
                item = idx

        self.AnnotationTypes.AnnotationType[item].Element.clear()
        del self.AnnotationTypes.AnnotationType[item]

    @property
    def nAnnotations(self):
        return self.Annotations.Annotation.__len__()

    @property
    def AnnotationIDs(self):
        return [Annotation['id'][0].param for Annotation in self.Annotations.Annotation]

    def add_Annotation(self, start_datetime:datetime, end_datetime:datetime, AnnotationTypeId=0, note='', type='GlobalAnnotation', channelName=''):
        # if AnnotationTypeId + '_aisc' in list(self.standard_UUID.keys()):
        #     AnnotationTypeId += '_aisc'

        if AnnotationTypeId + '_best' in list(self.standard_UUID.keys()):
            AnnotationTypeId += '_best'

        start_s = start_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")
        end_s = end_datetime.strftime("%Y-%m-%dT%H:%M:%S.%f")

        Annotation = self.Annotations.add_child(myXML_Annotation)
        Annotation.attributes[ET.QName(xsi, "type")] = type
        ID = Annotation.add_child(myXML_Id, param=str(uuid.uuid4()))

        startTimeUtc = Annotation.add_child(myXML_StartTimeUtc, param=start_s)
        endTimeUtc = Annotation.add_child(myXML_endTimeUtc, param=end_s)

        if isinstance(AnnotationTypeId, str):
            is_AType_set = False
            for idx, ATypepName in enumerate(self.AnnotationTypeKeys):
                if ATypepName == AnnotationTypeId:
                    AnnotationTypeId = idx
                    is_AType_set = True

            if is_AType_set is False:
                raise KeyError('AnnotationType cannot be assigned to an AnnotationGroup \"' + AnnotationTypeId + '\". Available AnnotationGroups \"' + str(self.AnnotationGroupKeys ))

        ATypeId_s = self.AnnotationTypeIDs[AnnotationTypeId]
        AnnotationTypeId = Annotation.add_child(myXML_AnnotationTypeId, param=ATypeId_s)

        if channelName:
            chName = Annotation.add_child(myXML_ChannelName, param=channelName)
            vertPosition = Annotation.add_child(myXML_VerticalPositionPercentage, param=0.25)

        if note.__len__() > 0:
            note = Annotation.add_child(myXML_Note, param=note)

    def remove_Annotation(self, item):
        if not (isinstance(item, int) and item >= 0 and item < self.Annotations.Annotation.__len__()) and not isinstance(item, str):
            raise KeyError('Annotation identifier must be an integer indexing Annotation groups or string representing a category name.')

        for idx, AnnotationTypeKey in enumerate(self.AnnotationTypesKeys):
            if AnnotationTypeKey == item:
                item = idx

        self.Annotations.Annotation[item].Element.clear()
        del self.Annotations.Annotation[item]

    def dump(self, path=None):
        def indent(elem, level=0): # creates new_line and indents in the xml file
            i = "\r\n" + level*"  "
            if len(elem):
                if not elem.text or not elem.text.strip():
                    elem.text = i + "  "
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
                for elem in elem:
                    indent(elem, level+1)
                if not elem.tail or not elem.tail.strip():
                    elem.tail = i
            else:
                if level and (not elem.tail or not elem.tail.strip()):
                    elem.tail = i

        AnnotationDataElement = deepcopy(self.AnnotationData.Element) # deep copy - indent not in the original file
        for idx, Element in enumerate(AnnotationDataElement):
            if Element.__len__() == 0:
                AnnotationDataElement[idx].clear()
                del AnnotationDataElement[idx]

        indent(AnnotationDataElement) # indents
        xml = ET.ElementTree(AnnotationDataElement) # Tree to write
        if isinstance(path, type(None)):
            path = self.AnnotationData._file_path

        xml.write(path, encoding='utf-8', xml_declaration=True) # dump file

def parse_CyberPSG_Annotations_xml(path):
    Annotations = CyberPSGFile(path)
    annotationTypes = Annotations.get_annotation_types()
    annotations = Annotations.get_annotations()
    dfAnnotations = pd.DataFrame(annotations)
    return dfAnnotations, annotationTypes











