# Copyright 2020-present, Mayo Clinic Department of Neurology - Bioelectronics Neurophysiology and Engineering Laboratory
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import os
# Check windows or linux and sets separator
if os.name == 'nt': DELIMITER = '\\'
else: DELIMITER = '/'



class myXML(ABC):
    """
    General function for XML Parsing utilizing xml.etree.ElementTree library and Elements of this lib.
    This function represents a single tag in XML structure.
    E.g.
    <meta attr1='0'>
    <time> '01-02-03' <\time>
    <\meta>
    etc.
    Object meta is defined by its tag 'meta' and and attribute 0. Has also the child with tag 'time'.
    Sub-object 'time' has only param '01-02-03'

    Object myXML is never meant to be called. Instead it is template for creating new objects inheriting from this instance.
    See bellow. Proper class definition is is:

    class myXML_Segment(myXML):
    tag = 'segment'
    ref_children = [myXML_url, myXML_Stop, myXML_Start, myXML_Id]

    tag is string exactly how it occurs in the XML file. ref_children is a list of objects which can be as sub-objects.
    This way, a proper structure can be defined and its integrity is ensured.
    If different tag appears than defined error is raised and structure must be defined.

    """

    tag = ''
    ref_children = []
    def __init__(self, Parent):
        """
        Init func which is NEVER used. Always use methods init_EmptyInstance or init_from_existing to initialize object myXML.
        """
        self._version = Parent._version
        self._file_id = Parent._file_id
        self._file_path = Parent._file_path

        self._children = {} # initialized in the init_from_existing func
        for ref_child in self.ref_children:
            self._children[ref_child.tag] = []

        for key in self._children.keys():
            setattr(self, key, self._children[key])

        self.attributes = myXML_AttributeHandler

    @classmethod
    def init_from_existing(cls, Parent, Element: ET.Element):
        """
        Main function to initialize instance from existing ET.Element which is read from the XML File.
        Inherits properties from Parent instance see __init__(). Element is a representation of a Tag using xml.etree lib.

        """
        iniClass = cls(Parent)
        # assert Element, 'Tag ' + iniClass.tag + 'is not declared in the processed file ' + Parent.file_path
        if not iniClass._set_Element(Element):
            raise AttributeError('Tag \'' + iniClass.tag + '\' is not declared in the processed file ' + Parent._file_path)

        iniClass.attributes = myXML_AttributeHandler(Element)
        iniClass._init_children()

        return iniClass

    @classmethod
    def init_EmptyInstance(cls, file_path = '', Parent = None, param = '', attributes = None):
        """
        Initialize empty instance

        """
        file_id = file_path.split('.')[-2].split(DELIMITER)[-1]

        if not Parent:
            class temp_class:
                def __init__(self, version='', file_id='', file_path=''):
                    self._version = version
                    self._file_id = file_id
                    self._file_path = file_path
                    self.tag = cls.tag
                    self.Element = ET.Element(cls.tag)

        Parent = temp_class(version="1.0", file_id=file_id, file_path=file_path)
        NewInstance = cls(Parent)
        NewInstance._set_Element(ET.Element(cls.tag))
        NewInstance.attributes = myXML_AttributeHandler(NewInstance.Element)
        NewInstance.param = param
        return NewInstance

    def add_child(self, child_initializer, param="", attributes={}):
        child = child_initializer(self)
        child.Element = ET.SubElement(self.Element, child.tag)
        child.attributes = myXML_AttributeHandler(child.Element)

        child.param = param
        for key in attributes.keys():
            child.attributes[key] = attributes[key]

        is_child_set = False
        for ref_child_key in self._children.keys():
            if ref_child_key == child.tag:
                self._children[child.tag].append(child)
                is_child_set = True

        if is_child_set is False:
            raise KeyError('Children \"' + child.tag + '\" cannot be assigned to a parent \"' + self.tag + '\". Defined children are: ' + str(self._children.keys()))

        return child

    def _init_children(self):
        """
        Initialize all children in a loop.
        """

        def _init_children_struct(ref_children):
            _children = {}
            for ref_child in ref_children:
                _children[ref_child.tag] = []
            return _children

        self._children = _init_children_struct(self.ref_children)
        for childEl in self.Element:
            self._init_child(childEl)

    def _init_child(self, ChildElement):
        """
        Inits a child from element

        """
        childInitializator = self._find_childClass(ChildElement.tag.split('}')[-1])
        assert childInitializator, 'Tag: ' + ChildElement.tag + \
                                              ' is not correctly declared in the structure' \
                                              ' of XML file. Please, have a look into the myXML' \
                                              'and add appropriate class and references into this package.'

        if type(self._children[ChildElement.tag.split('}')[-1]]) is list and self._children[ChildElement.tag.split('}')[-1]].__len__() == 0:
            self._children[ChildElement.tag.split('}')[-1]] = childInitializator.init_from_existing(self, ChildElement)
        elif not (type(self._children[ChildElement.tag.split('}')[-1]]) is list):
            self._children[ChildElement.tag.split('}')[-1]] = [self._children[ChildElement.tag.split('}')[-1]],
                                                childInitializator.init_from_existing(self, ChildElement)
                                                ]
        else:
            self._children[ChildElement.tag.split('}')[-1]].append(
                childInitializator.init_from_existing(self, ChildElement)
            )

    def _find_childClass(self, tag):
        for childClass in self.ref_children:
            if childClass.tag == tag:
                return childClass
        return False

    def _set_Element(self, aElement: ET.Element):
        if self._verify_input(aElement):
            self.Element = aElement
            return True
        return False

    def _verify_input(self, Element: ET.Element):
        ETag = Element.tag
        ETag = ETag.split('}')[-1]
        return ETag == self.tag

    def __getitem__(self, key):
        assert type(key) is str, 'Key to access child tags must be string!'
        assert key in self.keys(), 'Key \'' + '\' is not present in the existing tags.'
        return self._children[key]

    def __len__(self):
        return self.Element.__len__()

    def keys(self):
        #assert  self._children, 'Variable (dict) \'_children\' was not initialized  yet.'
        if self._children:
            return self._children.keys()
        return ''

    @property
    def has_children(self):
        return self.Element.__len__()

    @property
    def _get_children(self):
        return self.Element.getchildren()

    def _add_child(self, tagClass):
        self.Element.append(tagClass.Element)

    @property
    def has_param(self):
        try:
            return not self.Element.text.isspace()
        except:
            return False

    @property
    def param(self):
        if self.has_param:
            return self.Element.text
        return False

    @param.setter
    def param(self, txt_val):
        if txt_val is not str:
            txt_val = str(txt_val)
        self.Element.text = txt_val

    @property
    def has_attributes(self):
        return self.Element.attrib.__len__()

    def __str__(self):
        myStr = 'Object myXML_' + self.tag[0].capitalize() + self.tag[0:] + '\n' + \
        'tag: ' + self.tag + '\n' + \
        'param: ' + str(self.param) + '\n' + \
        'attributes: ' + self.attributes.__str__() + '\n'+ \
        'children: \n   '

        for key in self._children.keys():
            myStr = myStr + 'key: \'' + key + '\' ('
            child = self._children[key]
            if type(child) is list:
                if child.__len__() == 0:
                    myStr = myStr + '0'
                else:
                    myStr = myStr + str(child.__len__())
            else:
                myStr = myStr + '1'

            myStr = myStr + ')\n   '

        return myStr

    def __repr__(self):
        return self.__str__()


class myXML_AttributeHandler:
    """
    This is just a handler like a dict for. This is included in the myXML objects as a myXML.attributes
    """
    def __init__(self, AttrElement: ET.Element):
        #if AttrElement:
       self.Element = AttrElement

    def __len__(self):
        return self.Element.attrib.__len__()

    def __getitem__(self, item):
        #assert self.Element, 'There are no existing attributes.'
        if type(item) is int:
            assert item < self.__len__(), 'Index ' + str(item) + ' is higher than number of attributes. ' \
                                                                 'Number of attributes is ' + str(self.__len__())
            return self.Element.items(item)

        if type(item) is str:
            assert item in self.keys(), 'Attribute \'' + item + '\' does not exist for this instance.'
            return self.Element.attrib[item]

        return False

    def __setitem__(self, key, value):
        assert type(key) is str or isinstance(key, ET.QName), "Key for inserting an attribute must be string"
        #assert type(value) is str, "Attribute for xml parsing must be string"
        if not type(value): value = str(value)
        #if self.Element:
        self.Element.set(key, value)

    def __delitem__(self, key):
        assert type(key) is str, "Key for inserting an attribute must be string"
        del self.Element.attrib[key]

    def __str__(self):
        return self.Element.attrib.__str__()

    def __repr__(self):
        return self.__str__()

    def keys(self):
        return self.Element.keys()