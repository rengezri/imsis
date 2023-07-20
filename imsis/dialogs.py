#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains methods that are used for graphical user interaction.
"""

import cv2 as cv
import sys
from PySide6 import QtWidgets, QtGui, QtCore
import imsis as ims
import numpy as np
import os
from functools import partial
import collections


class Window(QtWidgets.QDialog):
    closed = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

    def closeEvent(self, event):
        self.closed.emit(self.windowTitle())
        super().closeEvent(event)

    def dialog_ok_cancel(self, title, text):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Question)
        msgBox.setText(text)
        msgBox.setWindowTitle(title)
        msgBox.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        msgBox.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)

        msgBox.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        ret = msgBox.exec()
        result = "Cancel"
        if ret == QtWidgets.QMessageBox.StandardButton.Ok:
            result = "Ok"
        return result

    def dialog_message(self, text):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msgBox.setText(text)
        msgBox.setWindowTitle('Message')
        msgBox.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        ret = msgBox.exec()

    def dialog_error(self, text):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msgBox.setText(text)
        msgBox.setWindowTitle('Error')
        msgBox.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        ret = msgBox.exec()

    def dialog_input_text(self, title, text, inputval):
        msgBox = QtWidgets.QInputDialog(self)
        ret, ok = msgBox.getText(self, title, text, text=inputval)
        return ret, ok

    def dialog_textbox(self, text, windowtext):
        msg = QtWidgets.QPlainTextEdit(self)
        msg.setWindowTitle(windowtext)
        msg.insertPlainText(text)
        msg.move(10, 10)
        msg.resize(512, 512)
        self.show()

    def dialog_textbox_html(self, text, windowtext):
        msg = QtWidgets.QTextEdit(self)
        msg.setWindowTitle(windowtext)
        msg.insertHtml(text)
        msg.move(10, 10)
        msg.resize(512, 512)
        self.show()


class PropertyGridWidget(QtWidgets.QDialog):
    _instances = set()  # Class level set to hold references to window instances

    def __init__(self,
                 properties,
                 text,
                 info):
        QtWidgets.QDialog.__init__(self)
        self._instances.add(self)  # Add the instance to the set

        self.properties = properties
        self.text = text
        self.info = info
        self.table = []

        # Grid Layout
        grid = QtWidgets.QGridLayout()

        self.setLayout(grid)
        self.setWindowTitle(self.text)
        newfont = QtGui.QFont("Sans Serif", 10)

        # Data
        data = self.properties

        # Create Empty 5x5 Table
        self.table = QtWidgets.QTableWidget(self)

        # QT6 changes
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)

        self.table.setRowCount(len(self.properties))
        self.table.setColumnCount(2)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)

        # Enter data onto Table
        horHeaders = []
        i = 0
        highlighted_index = 0

        self.setFont(newfont)
        for item in data:
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(item[0]))
            self.table.item(i, 0).setFlags(QtCore.Qt.ItemFlag.ItemIsEditable)

            if (item[1] == 'True' or item[1] == 'False'):
                checkbox = QtWidgets.QCheckBox()
                if item[1] == 'True':
                    checkbox.setChecked(True)

                else:
                    checkbox.setChecked(False)
                self.table.setCellWidget(i, 1, checkbox)

                checkbox.isChecked()
                checkbox.clicked.connect(self.onCheckBoxStateChanged)
            else:
                if (item[1][0] == '{' and item[1][-1] == '}'):
                    combo = QtWidgets.QComboBox()

                    # set combo color style

                    # cbstyle = "QComboBox QAbstractItemView {"
                    # cbstyle += " border: 1px solid grey;"
                    # cbstyle += " background: white;"
                    # cbstyle += " selection-background-color: blue;"
                    # cbstyle += " }"
                    cbstyle = " QComboBox {"
                    cbstyle += " background-color: white;"
                    cbstyle += " font: 12px;"
                    cbstyle += "}"

                    dct = eval(item[1])
                    combolistkeys = dct.keys()
                    combolistvals = dct.values()
                    highlighted_index = Dialogs.dialog_dictionary_activeindex(dct)

                    combo_box_options = combolistkeys
                    combo.setStyleSheet(cbstyle)

                    for t in combo_box_options:
                        combo.addItem(t)
                    self.table.setCellWidget(i, 1, combo)
                    combo.setCurrentIndex(highlighted_index)

                    combo.currentIndexChanged.connect(self.onComboIndexChanged)
                    # combo.currentIndexChanged.connect(onCurrentIndexChanged)

                else:
                    self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(item[1]))
            i = i + 1

        # print(highlighted_index)

        # Add Header
        self.table.setHorizontalHeaderLabels(horHeaders)

        # Adjust size of Table, first column adjusted to text, 2nd column adjusted to max 400
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        w0 = self.table.columnWidth(0)
        w1 = self.table.columnWidth(1)
        w0 = int(w0 * 1.1)
        w1 = int(w1 * 1.2)
        # print(w0,w1)
        if w0 > 400:
            w0 = 400
        if w1 > 400:
            w1 = 400
        self.table.setColumnWidth(0, w0)
        self.table.setColumnWidth(1, w1)

        # Add Table to Grid
        grid.addWidget(self.table, 0, 0)

        lines = len(data)
        if (lines > 25):
            lines = 25
        newheight = 100 + lines * 28

        Qinfo = QtWidgets.QLabel(self.info)

        okButton = QtWidgets.QPushButton("OK")
        if len(self.info) != 0:
            grid.addWidget(Qinfo)
        grid.addWidget(okButton)
        # self.setGeometry(200, 200, 600, newheight)

        # table.setSizeAdjustPolicy(
        #    QtWidgets.QAbstractScrollArea.AdjustToContents)

        self.table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContentsOnFirstShow)

        # self.raise_()
        okButton.clicked.connect(self.onButtonClicked)
        self.show()

    def onCheckBoxStateChanged(self):
        ch = self.sender()
        ix = self.table.indexAt(ch.pos())
        print(ix.row(), ix.column(), ch.isChecked())
        checkboxval = ch.isChecked()
        checkbox = QtWidgets.QCheckBox()
        # table.setItem(ix.row(),1,checkbox.setChecked(checkboxval))
        row = ix.row()
        self.properties[row][1] = str(checkboxval)

        self.table.viewport().update()  # Force the table to redraw its content

        # table.setItem[ix.row(),1,"True"]
        # table.setCellWidget(ix.row(), 1, setchecked(checkboxval))

    def onComboIndexChanged(self, value):
        row = self.table.currentRow()
        print(row)
        dct = eval(self.properties[row][1])
        actkey = list(dct.keys())[value]

        for key, val in dct.items():
            dct[key] = False
            if key == actkey:
                dct[key] = True

        self.properties[row][1] = str(dct)

    def onButtonClicked(self):
        allRows = self.table.rowCount()
        print("")
        print("start")
        for row in range(0, allRows):
            try:
                twi0 = self.table.item(row, 0).text()
                twi1 = self.table.item(row, 1).text()
                self.properties[row][0] = twi0
                self.properties[row][1] = twi1
            except:
                pass  # skip bool

        print("end")
        print("")
        self.choices = self.properties
        self.close()


class MultiButtonWidget(QtWidgets.QDialog):
    _instances = set()

    def __init__(self, listbuttons, name, info):
        QtWidgets.QDialog.__init__(self)
        self._instances.add(self)

        newfont = QtGui.QFont("Sans Serif", 10)
        layout = QtWidgets.QVBoxLayout(self)
        self.buttonpressed = 0
        self.buttons = []
        self.name = name

        for key, value in listbuttons.items():
            self.buttons.append(QtWidgets.QPushButton(key, self))
            self.buttons[-1].clicked.connect(partial(self.handleButton, data=value))
            if value == 0:
                self.buttons[-1].setEnabled(False)
            self.setFont(newfont)
            layout.addWidget(self.buttons[-1])
        self.setWindowTitle(self.name)

        Qinfo = QtWidgets.QLabel(info)
        if len(info) != 0:
            layout.addWidget(Qinfo)

    def handleButton(self, data="\n"):
        # print (data)
        self.buttonpressed = data
        # sys.exit(0) #sysexit is slow, while close is fast
        self.close()


class CheckListWidget(QtWidgets.QDialog):
    def __init__(self, name, datalist, info):
        super().__init__()

        self.name = name
        self.model = QtGui.QStandardItemModel()
        self.listView = QtWidgets.QListView()
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        for string, datatype in datalist:
            item = QtGui.QStandardItem(string)
            item.setCheckable(True)
            check = datatype
            if (check == True):
                item.setCheckState(QtCore.Qt.CheckState.Checked)
            else:
                item.setCheckState(QtCore.Qt.CheckState.Unchecked)
            self.model.appendRow(item)

        self.listView.setModel(self.model)
        Qinfo = QtWidgets.QLabel(info)
        self.okButton = QtWidgets.QPushButton('OK')
        self.abortButton = QtWidgets.QPushButton('Abort')
        self.selectButton = QtWidgets.QPushButton('Select All')
        self.unselectButton = QtWidgets.QPushButton('Unselect All')

        newfont = QtGui.QFont("Sans Serif", 10)
        self.setFont(newfont)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.abortButton)
        hbox.addWidget(self.selectButton)
        hbox.addWidget(self.unselectButton)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.listView)
        if len(info) != 0:
            vbox.addWidget(Qinfo)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setWindowTitle(self.name)
        self.okButton.clicked.connect(self.onAccepted)
        self.abortButton.clicked.connect(self.reject)
        self.selectButton.clicked.connect(self.select)
        self.unselectButton.clicked.connect(self.unselect)

    def onAccepted(self):
        itemlist = []
        for i in range(self.model.rowCount()):
            if self.model.item(i).checkState() == QtCore.Qt.CheckState.Checked:
                itemlist.append((self.model.item(i).text(), True))
            else:
                itemlist.append((self.model.item(i).text(), False))
        self.choices = collections.OrderedDict(itemlist)
        self.accept()

    def onAbort(self):
        self.choices = None
        self.reject()

    def select(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(QtCore.Qt.Checked)

    def unselect(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(QtCore.Qt.Unchecked)


class RadioButtonListWidget(QtWidgets.QDialog):
    _instances = set()

    def __init__(self, name, datalist, info):
        QtWidgets.QDialog.__init__(self)
        self._instances.add(self)

        self.name = name
        self.vbox = QtWidgets.QVBoxLayout()
        self.button_group = QtWidgets.QButtonGroup()

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        vbox = QtWidgets.QVBoxLayout(self)

        i = 0
        anybuttonselected = False
        for string, datatype in datalist:
            if datatype == True:
                anybuttonselected = True
        if anybuttonselected == False:
            datalist[0][1] = True

        for string, datatype in datalist:
            self.button_name = QtWidgets.QRadioButton("{}".format(string))
            self.button_group.addButton(self.button_name)
            self.button_group.setId(self.button_name, i)
            self.button_name.setChecked(datatype)
            vbox.addWidget(self.button_name)
            i = i + 1

        Qinfo = QtWidgets.QLabel(info)

        self.okButton = QtWidgets.QPushButton('OK')
        self.abortButton = QtWidgets.QPushButton('Abort')

        newfont = QtGui.QFont("Sans Serif", 10)
        self.setFont(newfont)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.abortButton)

        if len(info) != 0:
            vbox.addWidget(Qinfo)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setWindowTitle(self.name)
        self.okButton.clicked.connect(self.onAccepted)
        self.abortButton.clicked.connect(self.reject)

    def onAccepted(self):
        self.choices = self.button_group.checkedId()
        self.accept()

    def onAbort(self):
        self.choices = None
        self.reject()


class ComboBoxWidget(QtWidgets.QDialog):
    _instances = set()  # Class level set to hold references to window instances

    def __init__(self,
                 name,
                 datalist,
                 info):
        QtWidgets.QDialog.__init__(self)
        self._instances.add(self)  # Add the instance to the set

        self.name = name

        # self.model = QtGui.QSta.QStandardItemModel()
        self.vbox = QtWidgets.QVBoxLayout()

        # self.listView = QtWidgets.QListView()
        self.button_group = QtWidgets.QComboBox()

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)  # stop qtimer assertion when closing
        vbox = QtWidgets.QVBoxLayout(self)

        i = 0
        highlighted_index = 0
        # check state of all buttons, prevent that all buttons are disabled, if multiple buttons are selected last one is enabled.
        anybuttonselected = False
        for string, datatype in datalist:
            if datatype == True:
                anybuttonselected = True
                highlighted_index = i
            i = i + 1
        if anybuttonselected == False:
            datalist[0][1] = True

        i = 0
        for string, datatype in datalist:
            self.button_group.addItem(string)
            # self.button_group.setId(self.button_name, i)
            # self.button_name.setChecked(datatype)
            # vbox.addWidget(self.button_group)

            i = i + 1
        vbox.addWidget(self.button_group)

        # highlighted_index=2
        # print('index', highlighted_index)
        self.button_group.setCurrentIndex(highlighted_index)

        Qinfo = QtWidgets.QLabel(info)

        self.okButton = QtWidgets.QPushButton('OK')
        self.abortButton = QtWidgets.QPushButton('Abort')

        newfont = QtGui.QFont("Sans Serif", 10)
        self.setFont(newfont)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.abortButton)

        if len(info) != 0:
            vbox.addWidget(Qinfo)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setWindowTitle(self.name)
        self.okButton.clicked.connect(self.onAccepted)
        self.abortButton.clicked.connect(self.reject)

    def onAccepted(self):
        self.choices = self.button_group.currentIndex()
        self.accept()

    def onAbort(self):
        self.choices = None
        self.reject()


class ImageListViewWidget(QtWidgets.QListWidget):
    _instances = set()  # Class level set to hold references to window instances
    dropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        QtWidgets.QListWidget.__init__(self, parent)
        self._instances.add(self)  # Add the instance to the set
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            # self.emit(QtCore.SIGNAL("dropped"), links)
            self.dropped.emit(links)

        else:
            event.ignore()


class ImageListViewForm(QtWidgets.QDialog):

    def __init__(self, parent=None, title="Drag and drop image dialog"):
        super(ImageListViewForm, self).__init__(parent)
        self.setWindowTitle(title)
        self.view = ImageListViewWidget(self)
        self.view.dropped.connect(self.pictureDropped)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.view)
        self.setLayout(self.layout)
        self.url_list = []

    def pictureDropped(self, l):
        for url in l:
            if os.path.exists(url):
                print(url)
                icon = QtGui.QIcon(url)
                pixmap = icon.pixmap(72, 72)
                icon = QtGui.QIcon(pixmap)
                item = QtWidgets.QListWidgetItem(url, self.view)
                item.setIcon(icon)
                item.setStatusTip(url)
                self.url_list.append(url)


class Dialogs(object):
    def __init__(self):
        print("Dialog init")
        self.window = None  # added

    # DIALOGS QT
    @staticmethod
    def dialog_ok_cancel(text, windowtext="Confirm", alwaysontop=True):
        """Dialog ask Ok or Cancel

        :Parameters: title, text
        :Returns: result
        """

        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        ret = window.dialog_ok_cancel(windowtext, text)
        return ret

    @staticmethod
    def dialog_checklist(properties, windowtext="Checklist", info="", alwaysontop=True):
        """Dialog checklist

        List of ordered dictionary checkboxes is used such as
        items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)]) #ordered

        :Parameters: OrderedDict
        :Returns: OrderedDict

        .. code-block:: python

            items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)]) #ordered
            ret = em.Dialogs.dialog_checklist(items)
            print(ret)

        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        datalist = []
        for key, val in properties.items():
            datalist.append([key, val])

        window = CheckListWidget(name=windowtext, datalist=datalist, info=info)

        if alwaysontop:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        result = window.exec_()
        if result == QtWidgets.QDialog.Accepted:
            return window.choices
        else:
            return None

    def dialog_dictionary_activekey(dct):
        """Dialog combobox activekey

        retrieve the active key from a combobox in dictionary form
        :Parameters: dictionary
        :Returns: key
        """
        for key, value in dct.items():
            if value == True:
                keyfinal = key
        return keyfinal

    def dialog_dictionary_activeindex(dct):
        """Dialog combobox activekey

        retrieve the active key from a combobox in dictionary form
        :Parameters: dictionary
        :Returns: key
        """
        index = 0
        valfinal = 0
        for key, value in dct.items():
            if value == True:
                valfinal = index
            index = index + 1
        return valfinal

    @staticmethod
    def dialog_comboboxlist(properties, windowtext="Checklist", info="", alwaysontop=True):
        """Dialog checklist

        List of ordered dictionary checkboxes is used such as
        items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)]) #ordered

        :Parameters: OrderedDict
        :Returns: OrderedDict

        .. code-block:: python

            items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)]) #ordered
            ret = em.Dialogs.dialog_checklist(items)
            print(ret)

        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        datalist = []
        for key, val in properties.items():
            datalist.append([key, val])

        window = ComboBoxWidget(name=windowtext, datalist=datalist, info=info)

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        window.show()

        result = window.exec()
        try:
            newval = window.choices
            itemlist = []
            i = 0
            for key, val in properties.items():
                if i == newval:
                    itemlist.append((key, True))
                else:
                    itemlist.append((key, False))
                i = i + 1
            table = collections.OrderedDict(itemlist)
            # self.choices = collections.OrderedDict(itemlist)
            # self.accept()

            print("Selection made.")
        except:
            # table = properties
            table = None
            print("Cancel")

        ret = table
        return ret

    @staticmethod
    def dialog_radiobuttonlist(properties, windowtext="Checklist", info="", alwaysontop=True):
        """Dialog checklist

        List of ordered dictionary checkboxes is used such as
        items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)]) #ordered

        :Parameters: OrderedDict
        :Returns: OrderedDict

        .. code-block:: python

            items = collections.OrderedDict([('Left', True), ('Right', False), ('Up', True)]) #ordered
            ret = em.Dialogs.dialog_checklist(items)
            print(ret)

        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        datalist = []
        for key, val in properties.items():
            datalist.append([key, val])

        window = RadioButtonListWidget(name=windowtext, datalist=datalist, info=info)

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        result = window.exec()
        if result == QtWidgets.QDialog.Accepted:
            newval = window.choices
            itemlist = []
            i = 0
            for key, val in properties.items():
                if i == newval:
                    itemlist.append((key, True))
                else:
                    itemlist.append((key, False))
                i = i + 1
            table = collections.OrderedDict(itemlist)
            print("Selection made.")
        else:
            table = None
            print("Cancel")

        return table

    @staticmethod
    def dialog_multiplebuttons(listbuttons, windowtext="ButtonList", info="", alwaysontop=True):
        """Dialog multiple buttons
        buttons can be enabled disabled by setting them to 0.

        :Parameters: OrderedDict
        :Returns: OrderedDict

        .. code-block:: python

            #buttons = {'Left': 1, 'Right': 2, 'Up': 3, 'Down': 4, 'Abort': 5} #unordered, >Python3.7->ordered.
            buttons = OrderedDict([('Left', 1), ('Right', 2), ('Up', 3), ('Down', 4), ('Abort', 5)]) #ordered
            ret = em.Dialogs.dialog_multiplebuttons(buttons)

        """

        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = MultiButtonWidget(listbuttons, windowtext, info)
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        window.show()
        window.exec()
        ret = window.buttonpressed
        return ret

    @staticmethod
    def dialog_input(text, input, windowtext="Input", alwaysontop=True):
        """Dialog input, enter a string, float, int or bool and return input result with same type.
        Dialog will loop until a compatible datatype is entered.

        :Parameters: title, text, default_input
        :Returns: input (str or float or int or bool)

        .. code-block:: python

            ret = em.Dialogs.dialog_input("input",'fill in a text',"Hello")
            print(ret,type(ret))
            ret = em.Dialogs.dialog_input("input",'fill in a float',15)
            print(ret,type(ret))

        """
        entry_type = type(input)

        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        Continue = False
        while Continue == False:
            ret, ok = window.dialog_input_text(windowtext, text, str(input))

            if (ok == False):
                ret = input
                Continue = True
            try:
                if (entry_type is bool):
                    res = False
                    if (ret.upper() == "TRUE"):
                        res = True
                    ret = res
                else:
                    if (entry_type is int):
                        ret = int(ret)
                    else:
                        if (entry_type is float):
                            ret = float(ret)
                        else:
                            if (entry_type is dict):
                                ret = eval(ret)
                            else:
                                ret = ret  # not float or int therefore making it string
                Continue = True
            except:
                print("Error, incorrect type {0} expected.", entry_type)
        return ret

    @staticmethod
    def message(text, alwaysontop=True):
        """Text message

        :Parameters: text
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        window.dialog_message(text)

    @staticmethod
    def textbox(text, windowtext="Textbox", alwaysontop=True):
        """Text message
        This is a multi-line textbox
        :Parameters: text
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = Window()
        window.dialog_textbox(text, windowtext)
        window.exec()

    @staticmethod
    def textbox_html(text, windowtext="Textbox", alwaysontop=True):
        """Text message
        This is a multi-line textbox, input is a HTML string
        :Parameters: text
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.dialog_textbox_html(text, windowtext)
        window.exec()

    @staticmethod
    def error(text, alwaysontop=True):
        """Error message

        :Parameters: text
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        window.dialog_error(text)

    @staticmethod
    def dialog_propertygrid(properties, windowtext='Properties', verbose=True, info="", alwaysontop=True):
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        data = [[key, str(val) if val != "" else " "] for key, val in properties.items()]
        datatypes = [type(val) for val in properties.values()]

        window = PropertyGridWidget(data, windowtext, info)
        window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint if alwaysontop else window.windowFlags())
        window.show()

        result = window.exec()

        propgridwidget = window.properties

        properties = {}
        st1 = ''

        for row, (twi0, twi1) in enumerate(propgridwidget):
            st1 = f'{st1} {twi0} {twi1}'

            if datatypes[row] is bool:
                properties[twi0] = twi1 == 'True'
            elif datatypes[row] is int:
                properties[twi0] = int(twi1) if twi1.isdigit() else float(twi1)
            elif datatypes[row] is float:
                properties[twi0] = float(twi1)
            elif datatypes[row] is dict:
                properties[twi0] = eval(twi1)
            else:
                properties[twi0] = twi1  # not float or int therefore making it string

        if verbose:
            print('types: ', st1)
            print('out: ', properties)
        return properties

    @staticmethod
    def openfile_dialog(path='/', windowtext='Open File Dialog', filter="Images (*.png *.jpg *.bmp *.tif *.tiff)",
                        alwaysontop=True):
        """
        Open file dialog

        :Parameters: path, text, filter, alwaysontop
        :Returns: path
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(window, windowtext, path, filter)
        return filename

    @staticmethod
    def savefile_dialog(path='/', windowtext='Save File Dialog', filter="Images (*.png *.jpg *.bmp *.tif *.tiff)",
                        alwaysontop=True):
        """
        Save file dialog

        :Parameters: path, text, filter, alwaysontop
        :Returns: path
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(window, windowtext, path, filter)
        return filename

    @staticmethod
    def openfolder_dialog(path='/', windowtext='Open File Dialog', alwaysontop=True):
        """
        Open folder dialog

        :Parameters: path, text
        :Returns: path
        """
        # workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        folder = QtWidgets.QFileDialog.getExistingDirectory(window, windowtext, path)
        return folder

    # IMAGE INTERACTION
    @staticmethod
    def select_singlepoint(img, windowtext="Select Point by double click"):
        """Select a single point in an image

        :Parameters: windowtext=name of form
        :Returns: (x0,y0)
        """
        '''
        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
            img0 = frame.copy()  # avoid drawing inside the copy
        else:
            img0 = img.copy()
        '''

        img0 = img.copy()
        fx = 1

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        refPt = []
        cropping = False
        frame = ims.Image.Convert.toRGB(img0)
        pntslist = []

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            global refPt

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            if event == cv.EVENT_LBUTTONDBLCLK:
                refPt = [(x, y)]
                # refPt.append((x, y))

                # draw a line from point 0 to 1
                # cv.circle(frame, refPt[0], 2, (0, 255, 0), 2)
                refPt2 = [(int(refPt[0][0] * 1 / fx), int(refPt[0][1] * 1 / fx))]

                pntslist.append(refPt2)
                cv.imshow(windowtext, frame)

        clone = frame.copy()
        cv.namedWindow(windowtext)
        cv.setMouseCallback(windowtext, click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow(windowtext, frame)
            key = cv.waitKey(1) & 0xFF

            if (len(pntslist) > 0):
                break

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                frame = clone.copy()
            # monitor escape
            elif key == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        # close all open windows
        cv.destroyAllWindows()

        try:
            pnt = pntslist[0][0]
        except:
            pnt = (0, 0)
        print("point: {0}".format(pntslist))
        return pnt

    @staticmethod
    def select_points(img, windowtext="Select Points"):
        """Select multiple points in an image

        :Parameters: windowtext=name of form
        :Returns: list of shapes [shapenumber][(x0,y0),(x1,y1)]

        .. code-block:: python

            pnts = em.Dialogs.select_areas(img, 'Input: Select Areas')
            print(pnts)

        """

        '''
        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
            img0 = frame.copy()  # avoid drawing inside the copy
        else:
            img0 = img.copy()
        '''

        img0 = img.copy()
        fx = 1

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        refPt = []
        cropping = False
        frame = ims.Image.Convert.toRGB(img0)
        pntslist = []

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            global refPt

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            if event == cv.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]
                # refPt.append((x, y))

                # draw a line from point 0 to 1
                cv.circle(frame, refPt[0], 2, (0, 255, 0), 2)
                refPt2 = [(int(refPt[0][0] * 1 / fx), int(refPt[0][1] * 1 / fx))]

                pntslist.append(refPt2)
                cv.imshow(windowtext, frame)

        clone = frame.copy()
        cv.namedWindow(windowtext)
        cv.setMouseCallback(windowtext, click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow(windowtext, frame)
            key = cv.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                frame = clone.copy()
            # monitor escape
            elif key == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        # close all open windows
        cv.destroyAllWindows()
        print("points: {0}".format(pntslist))
        return pntslist

    @staticmethod
    def select_lines(img, windowtext="Select Lines"):
        """Draw multiple lines in an image and return (x0,y0,x1,y1) for each rectangle

        :Parameters: windowtext=name of form
        :Returns: list of shapes [shapenumber][(x0,y0),(x1,y1)]
        :rtype: object
        """

        img0 = img.copy()

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        refPt = []
        img0 = ims.Image.Convert.toRGB(img0)

        pntslist = []

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            global refPt

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            if event == cv.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]

            # check to see if the left mouse button was released
            elif event == cv.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))

                # draw the final line from point 0 to 1
                cv.line(img0, refPt[0], refPt[1], (0, 255, 0), 2)

                refPt2 = [(int(refPt[0][0] * 1), int(refPt[0][1] * 1)),
                          (int(refPt[1][0] * 1), int(refPt[1][1] * 1))]
                pntslist.append(refPt2)
                cv.imshow(windowtext, img0)

            # check to see if the mouse is moving and the left mouse button is down
            elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
                # draw a temporary line from point 0 to the current mouse position
                tempImg = img0.copy()
                cv.line(tempImg, refPt[0], (x, y), (0, 255, 0), 2)
                cv.imshow(windowtext, tempImg)

            # clone = img0.copy()

        cv.setMouseCallback(windowtext, click_and_crop)
        cv.imshow(windowtext, img0)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            key = cv.waitKey(1) & 0xFF

            # monitor escape
            if key == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        # close all open windows
        cv.destroyAllWindows()
        print("lines: {0}".format(pntslist))
        return pntslist

    @staticmethod
    def select_areas(img, windowtext="Select Areas"):
        """
        Draw multiple rectangles in an image and return the position of the rectangles

        :Parameters: windowtext=name of form
        :Returns: list of shapes [shapenumber][(x0,y0),(x1,y1)]
        """

        img0 = img.copy()

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        refPt = []
        img0 = ims.Image.Convert.toRGB(img0)

        pntslist = []

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            global refPt

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            if event == cv.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]

            # check to see if the left mouse button was released
            elif event == cv.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))

                # draw the final line from point 0 to 1
                cv.rectangle(img0, refPt[0], refPt[1], (0, 255, 0), 2)

                refPt2 = [(int(refPt[0][0] * 1), int(refPt[0][1] * 1)),
                          (int(refPt[1][0] * 1), int(refPt[1][1] * 1))]
                pntslist.append(refPt2)
                cv.imshow(windowtext, img0)

            # check to see if the mouse is moving and the left mouse button is down
            elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
                # draw a temporary line from point 0 to the current mouse position
                tempImg = img0.copy()
                cv.rectangle(tempImg, refPt[0], (x, y), (0, 255, 0), 2)

                cv.imshow(windowtext, tempImg)

        cv.setMouseCallback(windowtext, click_and_crop)
        cv.imshow(windowtext, img0)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            key = cv.waitKey(1) & 0xFF

            # monitor escape
            if key == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        # close all open windows
        cv.destroyAllWindows()
        print("areas: {0}".format(pntslist))
        return pntslist

    @staticmethod
    def adjust_mask(img, windowtext="Select Mask"):
        """create an image mask by setting the intensity range and blur. Returns: image, min,max,blur

        :Parameters: windowtext=name of form
        :Returns: ImageOut, Min,Max,Blur

        .. code-block:: python

            print('adjust_mask')
            thresh1, min,max,blur = em.Dialogs.adjust_mask(img,'Select Image mask')
            print(min,max,blur)
            em.View.plot(thresh1,'')
        """

        def subfunction(img, min, max, blur):
            imout = cv.GaussianBlur(img, (blur, blur), 0)
            thresh1 = cv.inRange(imout, min, max)
            return thresh1

        def nothing(x):
            pass

        img = ims.Image.Convert.to8bit(img)
        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("Min", windowtext, 32, 255, nothing)
        cv.createTrackbar("Max", windowtext, 128, 255, nothing)
        cv.createTrackbar("Blur", windowtext, 1, 50, nothing)

        while (1):
            min = cv.getTrackbarPos("Min", windowtext)
            max = cv.getTrackbarPos("Max", windowtext)
            blur = cv.getTrackbarPos("Blur", windowtext)
            if (blur % 2 == 0):
                blur = blur + 1
            thresh1 = subfunction(frame, min, max, blur)

            cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()
        print("Mask: thresholdedimage, Min={},Max={},Blur={}".format(min, max, blur))
        thresh1 = subfunction(img, min, max, blur)
        return thresh1, min, max, blur

    @staticmethod
    def adjust_mask_with_overlay(img, windowtext="Select Mask", text="Zoom +/-/r Hide h"):
        """create an image mask by setting the intensity range and blur. Returns: image, min,max,blur
        zoom in/out with +/-, reset zoom with r
        hide/unhide mask with h
        :Parameters: image, windowtext=name of form
        :Returns: Image_Thresholded, Min,Max,Blur
        """

        def subfunction(img, min, max, blur, zoomfactor, hidemask):
            img = cv.resize(img, None, fx=zoomfactor, fy=zoomfactor)
            imout = cv.GaussianBlur(img, (blur, blur), 0)
            thresh0 = cv.inRange(imout, min, max)
            thresh1 = ims.Image.Process.merge2channels(thresh0, img)
            if hidemask == True:
                thresh1 = img
            if text:
                thresh1 = ims.Analyze.add_text(thresh1, 0, 0, text, fontsize=20)

            return thresh1

        def subfunction_final(img, min, max, blur):
            imout = cv.GaussianBlur(img, (blur, blur), 0)
            thresh1 = cv.inRange(imout, min, max)
            return thresh1

        def nothing(x):
            pass

        img = ims.Image.Convert.to8bit(img)
        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("Min", windowtext, 32, 255, nothing)
        cv.createTrackbar("Max", windowtext, 128, 255, nothing)
        cv.createTrackbar("Blur", windowtext, 1, 50, nothing)

        zoomfactor = 1
        hidemask = False

        minlast = 0
        maxlast = 0
        blurlast = 0

        while (1):
            min = cv.getTrackbarPos("Min", windowtext)
            max = cv.getTrackbarPos("Max", windowtext)
            blur = cv.getTrackbarPos("Blur", windowtext)
            if (blur % 2 == 0):
                blur = blur + 1

            if (min != minlast) or (max != maxlast) or (blur != blurlast):
                minlast = min
                maxlast = max
                blurlast = blur
                thresh1 = subfunction(frame, min, max, blur, zoomfactor, hidemask)

            cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
            if k == 43:
                zoomfactor = zoomfactor * 1.25
            if k == 45:
                zoomfactor = zoomfactor * 0.75
            if k == 114:
                zoomfactor = 1
            if k == 104:
                if hidemask == True:
                    hidemask = False
                else:
                    hidemask = True

        cv.destroyAllWindows()
        print("MaskWithBackground: thresholdedimage, Min={},Max={},Blur={}".format(min, max, blur))
        # thresh1 = subfunction(img, min, max, blur, zoomfactor=1, hidemask=False)
        thresh1 = subfunction_final(img, min, max, blur)
        return thresh1, min, max, blur

    @staticmethod
    def adjust_contours_after_masking(img, min, max, blur, windowtext="Select Mask", text="Zoom +/-/r Hide h"):
        """Adjust the settings for contours after masking
        zoom in/out with +/-, reset zoom with r
        hide/unhide mask with h
        :Parameters: image, windowtext=name of form
        :Returns: Image_Thresholded, Min,Max,Blur
        """

        def subfunction(img, min, max, blur, minArea, maxArea, dt, zoomfactor, hidemask):
            if minArea >= maxArea:
                minArea = maxArea - 1  # avoid division by 0
            # img = cv.resize(img, None, fx=zoomfactor, fy=zoomfactor)
            imout = cv.GaussianBlur(img, (blur, blur), 0)
            thresh0 = cv.inRange(imout, min, max)
            # thresh1 = ims.Image.Process.Colormap.falsecolor_merge2channels(thresh0, img)
            fn = 'overlay.png'
            # print(min, max, blur, minArea, maxArea, dt)
            try:
                overlay, labels, markers, featurelist = ims.Analyze.FeatureProperties.get_featureproperties(img,
                                                                                                            thresh0,
                                                                                                            minarea=minArea,
                                                                                                            maxarea=maxArea,
                                                                                                            applydistancemap=True,
                                                                                                            distance_threshold=dt / 100)
                overlay2 = ims.Analyze.FeatureProperties.get_image_with_ellipses(overlay, featurelist)
            except:
                print("Error")
                overlay2 = img.copy()

            return overlay2

        def subfunction_final(img, min, max, blur, minArea, maxArea, dt):
            # print("final ", min, max, blur, minArea, maxArea, dt)
            imout = cv.GaussianBlur(img, (blur, blur), 0)
            thresh0 = cv.inRange(imout, min, max)
            overlay, labels, markers, featurelist = ims.Analyze.FeatureProperties.get_featureproperties(img,
                                                                                                        thresh0,
                                                                                                        minarea=minArea,
                                                                                                        maxarea=maxArea,
                                                                                                        applydistancemap=True,
                                                                                                        distance_threshold=dt / 100)
            return overlay

        def nothing(x):
            pass

        def resizeimg(img):
            fx = 1024 / img.shape[1]
            if fx < 1:
                frame = cv.resize(img, None, fx=fx, fy=fx)
            else:
                frame = img.copy()
            return frame

        img = ims.Image.Convert.to8bit(img)
        frame = img.copy()
        orig = img.copy()
        maxarealimit = int(img.shape[0] * img.shape[1] * 0.1)

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("MinArea", windowtext, 200, maxarealimit, nothing)
        cv.createTrackbar("MaxArea", windowtext, 1500, maxarealimit, nothing)
        cv.createTrackbar("DistanceThreshold", windowtext, 1, 100, nothing)

        zoomfactor = 1
        hidemask = False

        minArealast = 0
        maxArealast = 0
        dtlast = 0

        while (1):
            minArea = cv.getTrackbarPos("MinArea", windowtext)
            maxArea = cv.getTrackbarPos("MaxArea", windowtext)
            dt = cv.getTrackbarPos("DistanceThreshold", windowtext)

            if (dt != dtlast) or (minArea != minArealast) or (maxArea != maxArealast):
                thresh1 = subfunction(frame, min, max, blur, minArea, maxArea, dt, zoomfactor, hidemask)
                minArealast = minArea
                maxArealast = maxArea
                dtlast = dt

            cv.imshow(windowtext, resizeimg(thresh1))
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
            if k == 43:
                zoomfactor = zoomfactor * 1.25
            if k == 45:
                zoomfactor = zoomfactor * 0.75
            if k == 114:
                zoomfactor = 1
            if k == 104:
                if hidemask == True:
                    hidemask = False
                else:
                    hidemask = True

        cv.destroyAllWindows()
        overlay = subfunction_final(orig, min, max, blur, minArea, maxArea, dt)
        print(
            "adjust_contours_after_masking: min {}, max {}, blur {}, minarea {}, maxarea {}, dt/100 {}, minarealast{}, maxarealast {}, dtlast {}".format(
                min, max, blur, minArea, maxArea, dt / 100, minArealast, maxArealast, dtlast))
        return overlay, minArea, maxArea, dt / 100

    @staticmethod
    def select_edges(img, windowtext="Select Edges"):
        """Find all edges in image. Returns: threshold image, threshold value,blur

        :Parameters: windowtext=name of form
        :Returns: ImageOut, Threshold, Blur

        .. code-block:: python

            print('adjust_mask')
            thresh1, min,max,blur = em.Dialogs.adjust_mask(img,'Select Image mask')
            print(min,max,blur)
            em.View.plot(thresh1,'')
        """

        def nothing(x):
            pass

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("NonMaxSupp", windowtext, 32, 255, nothing)
        cv.createTrackbar("Blur", windowtext, 1, 50, nothing)

        while (1):
            min = cv.getTrackbarPos("NonMaxSupp", windowtext)
            blur = cv.getTrackbarPos("Blur", windowtext)
            if (blur % 2 == 0):
                blur = blur + 1
            thresh1, angle = ims.Image.Process.gradient_image_nonmaxsuppressed(frame, blur, min)

            cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break

        cv.destroyAllWindows()

        thresh1, angle = ims.Image.Process.gradient_image_nonmaxsuppressed(img, blur, min)
        print("Select edges: ThresholdedImage, min={}, blur={}".format(min, blur))
        return thresh1, min, blur

    @staticmethod
    def adjust_FD_bandpass_filter(img, windowtext="Apply BandPass Filter in Frequency Domain"):
        """Apply a bandpass filter in frequency domain

        :Parameters: windowtext=name of form
        :Returns: filtered, mask, bandcenter,bandwidth

        """

        def nothing(x):
            pass

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("BandCenter", windowtext, 32, 255, nothing)
        cv.createTrackbar("BandWidth", windowtext, 16, 255, nothing)
        cv.createTrackbar("lpType", windowtext, 0, 2, nothing)

        while (1):
            bandcenter = cv.getTrackbarPos("BandCenter", windowtext)
            bandwidth = cv.getTrackbarPos("BandWidth", windowtext)
            lptype = cv.getTrackbarPos("lpType", windowtext)

            filtered, mask = ims.Image.Process.FD_bandpass_filter(img, bandcenter, bandwidth, bptype=lptype)
            # combined = mask
            combined = ims.Image.Tools.concat_two_images(filtered, mask)

            # ims.View.plot(combined)
            # sys.exit()
            # combined = filtered
            cv.imshow(windowtext, combined)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()

        filtered, mask = ims.Image.Process.FD_bandpass_filter(img, bandcenter, bandwidth, bptype=lptype)
        print("FFT BandPass bandcenter={}, bandwidth={}, lpType={}".format(bandcenter, bandwidth, lptype))
        return filtered, mask, bandcenter, bandwidth, lptype

    @staticmethod
    def adjust_HSL(img, windowtext="HSL Channels"):
        """Adjust HSL Channels dialog: (Hue, Saturation, Lightness) of a color image. Returns image, h,s,l

        :Parameters: image, windowtext
        :Returns: image, hue, saturation, lightness
        """

        # Note: for color map images additional casting may be required

        def nothing(x):
            pass

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("Hue", windowtext, 0, 180, nothing)
        cv.createTrackbar("Sat", windowtext, 255, 255 * 2, nothing)
        cv.createTrackbar("Light", windowtext, 255, 255 * 2, nothing)

        while (1):
            Hue = cv.getTrackbarPos("Hue", windowtext)
            Sat = cv.getTrackbarPos("Sat", windowtext)
            Light = cv.getTrackbarPos("Light", windowtext)

            hsl = ims.Image.Adjust.adjust_HSL(frame, Hue, Sat - 255, Light - 255)
            cv.imshow(windowtext, hsl)

            # cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()
        print("Adjust HSL: image, Hue={},Sat={},Light={}".format(Hue, Sat - 255, Light - 255))
        hls = ims.Image.Adjust.adjust_HSL(img, Hue, Sat - 255, Light - 255)
        return hls, Hue, Sat, Light

    @staticmethod
    def adjust_HSV(img, windowtext="HSV Channels"):
        """Adjust HSV Channels dialog: (Hue, Saturation, Value) of a color image. Returns image, h,s,v

        :Parameters: image, windowtext
        :Returns: image, hue, saturation, lightness
        """

        # Note: for color map images additional casting may be required

        def nothing(x):
            pass

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("Hue", windowtext, 0, 180, nothing)
        cv.createTrackbar("Sat", windowtext, 255, 255 * 2, nothing)
        cv.createTrackbar("Value", windowtext, 255, 255 * 2, nothing)

        while (1):
            Hue = cv.getTrackbarPos("Hue", windowtext)
            Sat = cv.getTrackbarPos("Sat", windowtext)
            Val = cv.getTrackbarPos("Value", windowtext)

            hsv = ims.Image.Adjust.adjust_HSV(frame, Hue, Sat - 255, Val - 255)
            cv.imshow(windowtext, hsv)

            # cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()
        print("Adjust HSV: image, Hue={},Sat={},Value={}".format(Hue, Sat - 255, Val - 255))
        hls = ims.Image.Adjust.adjust_HSV(img, Hue, Sat - 255, Val - 255)
        return hls, Hue, Sat, Val

    @staticmethod
    def adjust_contrast_brightness_gamma(img, windowtext="Contrast Brightness and Gamma Adjustment"):
        """Adjust Contrast Brightness Gamma
        #Brightness value range -255 to 255
        #Contrast value range -127 to 127
        #adjust gamma [0..3.0]

        :Parameters: image, windowtext
        :Returns: image, contrast, brightness, gamma
        """

        def nothing(x):
            pass

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        c, b, g = 0, 0, 1

        cv.createTrackbar("Contrast", windowtext, 127, 127 * 2, nothing)
        cv.createTrackbar("Brightness", windowtext, 255, 255 * 2, nothing)
        cv.createTrackbar("Gamma", windowtext, 84, 254, nothing)

        while (1):
            c = cv.getTrackbarPos("Contrast", windowtext)
            b = cv.getTrackbarPos("Brightness", windowtext)
            g = cv.getTrackbarPos("Gamma", windowtext)
            c = c - 127
            b = b - 255
            g = (g + 1) / 85

            out = ims.Image.Adjust.adjust_contrast_brightness(frame, c, b)
            out = ims.Image.Adjust.adjust_gamma(out, g)
            cv.imshow(windowtext, out)

            # cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()
        out = ims.Image.Adjust.adjust_contrast_brightness(img, c, b)
        out = ims.Image.Adjust.adjust_gamma(out, g)
        print("ContrastBrightnessGamma: image, Contrast={},Brightness={},Gamma={}".format(c, b, g))

        return out, c, b, g

    @staticmethod
    def adjust_blending(img0, img1, windowtext="Image Blending"):
        """Adjust mixing level by a slider
        #Brightness value range 0..100

        :Parameters: image, windowtext
        :Returns: image, contrast, brightness, gamma
        """

        def nothing(x):
            pass

        fx = 1024 / img0.shape[1]
        if fx < 1:
            frame0 = cv.resize(img0, None, fx=fx, fy=fx)
            frame1 = cv.resize(img1, None, fx=fx, fy=fx)
        else:
            frame0 = img0.copy()
            frame1 = img1.copy()

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("Alpha", windowtext, 0, 100, nothing)

        while (1):
            alpha = cv.getTrackbarPos("Alpha", windowtext)

            a = img0
            b = img1
            gamma = 1
            beta = 1 - alpha / 100
            out = cv.addWeighted(a, alpha / 100, b, beta, gamma)
            cv.imshow(windowtext, out)

            # cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()

        a = img0
        b = img1
        gamma = 1
        beta = 1 - alpha / 100
        out = cv.addWeighted(a, alpha / 100, b, beta, gamma)

        print("Alpha: image, {}".format(alpha))

        return out, alpha

    @staticmethod
    def adjust_openclose(img, windowtext="Open Close Erode Dilate Adjustment"):
        """
        Adjust morphological operations Open Close Erode Dilate

        :Parameters: windowtext=name of form
        :Returns: image

        .. code-block:: python

            print('Adjust Morphological operations')
            img0 = em.Image.Convert.toRGB(thresh1)
            thresh2 = em.Dialogs.adjust_openclose(img0,'Morphological operations')
            em.View.plot(thresh2,'')
        """

        def nothing(x):
            pass

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        cv.createTrackbar("Close", windowtext, 1, 15, nothing)
        cv.createTrackbar("Open", windowtext, 1, 15, nothing)
        cv.createTrackbar("Dilate", windowtext, 1, 15, nothing)
        cv.createTrackbar("Erode", windowtext, 1, 15, nothing)

        while (1):
            cval = cv.getTrackbarPos("Close", windowtext)
            oval = cv.getTrackbarPos("Open", windowtext)
            dval = cv.getTrackbarPos("Dilate", windowtext)
            eval = cv.getTrackbarPos("Erode", windowtext)

            if (cval < 1):
                cval = 1
            if (oval < 1):
                oval = 1
            if (dval < 1):
                dval = 1
            if (eval < 1):
                eval = 1

            se1 = cv.getStructuringElement(cv.MORPH_RECT, (cval, cval))
            se2 = cv.getStructuringElement(cv.MORPH_RECT, (oval, oval))
            se4 = cv.getStructuringElement(cv.MORPH_RECT, (dval, dval))
            se3 = cv.getStructuringElement(cv.MORPH_RECT, (eval, eval))
            mask = cv.morphologyEx(frame, cv.MORPH_CLOSE, se1)
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se2)
            mask = cv.morphologyEx(mask, cv.MORPH_ERODE, se3)
            thresh1 = cv.morphologyEx(mask, cv.MORPH_DILATE, se4)

            cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()

        mask = cv.morphologyEx(img, cv.MORPH_CLOSE, se1)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se2)
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, se3)
        thresh1 = cv.morphologyEx(mask, cv.MORPH_DILATE, se4)
        print("OpenCloseErodeDilate: Image")
        return thresh1

    # IMAGE INTERACTION
    @staticmethod
    def image_view(img, windowtext="Image View"):
        """Show Image no interaction

        :Parameters: image, windowtext=name of form
        """

        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
        else:
            frame = img.copy()
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        frame = ims.Image.Convert.toRGB(frame)

        clone = frame.copy()
        cv.namedWindow(windowtext)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow(windowtext, frame)
            key = cv.waitKey(1) & 0xFF

            if key == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        # close all open windows
        cv.destroyAllWindows()

    # IMAGE INTERACTION
    @staticmethod
    def image_compare(img0, img1, windowtext="Image Compare"):
        """
        Compare 2 images using a slider

        :Parameters: image0, image1, windowtext=name of form

        """

        def nothing(x):
            pass

        fx = 1024 / img0.shape[1]
        if fx < 1:
            frame0 = cv.resize(img0, None, fx=fx, fy=fx)
            frame1 = cv.resize(img1, None, fx=fx, fy=fx)
        else:
            frame0 = img0.copy()
            frame1 = img1.copy()

        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        # img0 = em.Image.Convert.toRGB(img0)
        fx1 = 1

        img0 = frame0.copy()
        img1 = frame1.copy()
        cval = int(img1.shape[1] / 2)
        img0[0:img0.shape[0], cval:img0.shape[1] + cval] = img1[0:img1.shape[0], cval:img1.shape[1]]
        cv.line(img0, (cval, 0), (cval, img0.shape[1]), (255, 255, 255), 2)
        cv.imshow(windowtext, img0)

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            global mdown, cval

            img0 = frame0.copy()
            img1 = frame1.copy()
            try:
                if event == cv.EVENT_LBUTTONDOWN:
                    mdown = True
                # check to see if the left mouse button was released
                elif event == cv.EVENT_LBUTTONUP:
                    mdown = False

                    # draw a line from point 0 to 1
                    # cv.line(img0, refPt[0], refPt[1], (0, 255, 0), 2)
                if mdown == True:
                    cval = x
                    if (cval < 1):
                        cval = 1
                    if (cval >= img1.shape[1]):
                        cval = img1.shape[1] - 1
                    img0[0:img0.shape[0], cval:img0.shape[1] + cval] = img1[0:img1.shape[0], cval:img1.shape[1]]
                    cv.line(img0, (cval, 0), (cval, img0.shape[1]), (255, 255, 255), 2)
                    cv.imshow(windowtext, img0)
            except:
                cval = int(img1.shape[1] / 2)
                img0[0:img0.shape[0], cval:img0.shape[1] + cval] = img1[0:img1.shape[0], cval:img1.shape[1]]
                cv.line(img0, (cval, 0), (cval, img0.shape[1]), (255, 255, 255), 2)
                cv.imshow(windowtext, img0)

        cv.setMouseCallback(windowtext, click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            # cv.imshow(windowtext, img0)
            key = cv.waitKey(1) & 0xFF

            # if key == ord("r"):
            #    fx1=1
            if key == 27:
                break
            try:
                if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                    break
            except:
                print("window already closed.")
                break
        cv.destroyAllWindows()

    @staticmethod
    def dialog_imagelistview(windowtext="Drag and drop image dialog", alwaysontop=True):
        """
        Image listview dialog. The listview allows for dragging dropping and thumbnail previewing of files.

        :Returns: list of urls
        """

        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = ImageListViewForm()

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowType.WindowStaysOnTopHint)

        window.exec()
        url_list = window.url_list
        return url_list

# main
