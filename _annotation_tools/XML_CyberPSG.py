# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from _annotation_tools.xml_structured_parsing.myXML import myXML, DELIMITER, ET

#def parser_xml_CyberPSG(path):
#    class EmptyClass:
#        def __init__(self, version=1.1, file_id='SPARSE_MIX', file_path=''):
#            self._version = version
#            self._file_id = file_id
#            self._file_path = file_path
#
#    PseudoParent = EmptyClass(version=1.1, file_id=path.split(DELIMITER)[-1].split('.')[0], file_path=path)
#    return myXML_AnnotationData.init_from_existing(PseudoParent, ET.parse(path).getroot())

##### VERSION #####
class myXML_Note(myXML):
    tag = 'note'

class myXML_Id(myXML):
    tag = 'id'

##### META #####
class myXML_Created(myXML):
    tag = 'created'

class myXML_Modified(myXML):
    tag = 'modified'

class myXML_Name(myXML):
    tag = 'name'

class myXML_AnnotationGroup(myXML):
    tag = 'AnnotationGroup'
    ref_children = [myXML_Id, myXML_Created, myXML_Modified, myXML_Name, myXML_Note]

class myXML_AnnotationGroups(myXML):
    tag = 'AnnotationGroups'
    ref_children = [myXML_AnnotationGroup]

class myXML_Description(myXML):
    tag = 'description'

class myXML_stdDurationInSec(myXML):
    tag = 'stdDurationInSec'

class myXML_startsWithEpoch(myXML):
    tag = 'startsWithEpoch'

class myXML_Hotkey(myXML):
    tag = 'hotkey'

class myXML_Color(myXML):
    tag = 'color'

class myXML_GroupAsociations(myXML):
    tag = 'groupAssociations'
    ref_children = [myXML_Id]

class myXML_AnnotationType(myXML):
    tag = 'AnnotationType'
    ref_children = [myXML_Id, myXML_Created, myXML_Modified, myXML_Name, myXML_Description, myXML_stdDurationInSec, myXML_startsWithEpoch, myXML_Hotkey, myXML_Color, myXML_GroupAsociations, myXML_Note]

class myXML_AnnotationTypes(myXML):
    tag = 'AnnotationTypes'
    ref_children = [myXML_AnnotationType]

class myXML_StartTimeUtc(myXML):
    tag = 'startTimeUtc'

class myXML_endTimeUtc(myXML):
    tag = 'endTimeUtc'

class myXML_AnnotationTypeId(myXML):
    tag = 'annotationTypeId'

class myXML_ChannelName(myXML):
    tag = 'channelName'

class myXML_VerticalPositionPercentage(myXML):
    tag = 'verticalPositionPercentage'

class myXML_Annotation(myXML):
    tag = 'Annotation'
    ref_children = [myXML_Id, myXML_Created, myXML_Modified, myXML_StartTimeUtc, myXML_endTimeUtc, myXML_AnnotationTypeId, myXML_ChannelName, myXML_VerticalPositionPercentage, myXML_Note]

class myXML_Annotations(myXML):
    tag = 'Annotations'
    ref_children = [myXML_Annotation]

##### 1st Level ######
class myXML_AnnotationData(myXML):
    tag = 'AnnotationData'
    ref_children = [myXML_AnnotationGroups, myXML_AnnotationTypes, myXML_Annotations]