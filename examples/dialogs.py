#!/usr/bin/env python

"""
This file is part of IMSIS
Licensed under the MIT license:
http://www.opensource.org/licenses/MIT-license

This module contains methods that are used for graphical user interaction.
"""

import cv2 as cv

import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import imsis as ims
import numpy as np
import os
from functools import partial
import copy
import collections

print("PyQt", QtCore.PYQT_VERSION_STR)


class Window(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

    def dialog_ok_cancel(self, title, text):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Question)
        msgBox.setText(text)
        msgBox.setWindowTitle(title)
        msgBox.addButton(QtWidgets.QMessageBox.Ok)
        msgBox.addButton(QtWidgets.QMessageBox.Cancel)

        msgBox.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        ret = msgBox.exec_()
        result = "Cancel"
        if ret == QtWidgets.QMessageBox.Ok:
            result = "Ok"
        return result

    def dialog_message(self, text):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
        msgBox.setText(text)
        msgBox.setWindowTitle('Message')
        msgBox.addButton(QtWidgets.QMessageBox.Ok)
        ret = msgBox.exec_()

    def dialog_error(self, text):
        msgBox = QtWidgets.QMessageBox(self)
        msgBox.setIcon(QtWidgets.QMessageBox.Critical)
        msgBox.setText(text)
        msgBox.setWindowTitle('Error')
        msgBox.addButton(QtWidgets.QMessageBox.Ok)
        ret = msgBox.exec_()

    def dialog_input_text(self, title, text, inputval):
        msgBox = QtWidgets.QInputDialog(self)
        ret, ok = msgBox.getText(self, title, text, text=inputval)
        return ret, ok

    def dialog_textbox(self, title, text):
        msg = QtWidgets.QPlainTextEdit(self)
        msg.insertPlainText(text)
        msg.move(10, 10)
        msg.resize(512, 512)

    def dialog_textbox_html(self, title, text):
        msg = QtWidgets.QTextEdit(self)
        msg.insertHtml(text)
        msg.move(10, 10)
        msg.resize(512, 512)

    def dialog_propertygrid(self, properties, text, info=""):

        def onCheckBoxStateChanged():
            ch = self.sender()
            ix = table.indexAt(ch.pos())
            print(ix.row(), ix.column(), ch.isChecked())
            checkboxval = ch.isChecked()
            checkbox = QtWidgets.QCheckBox()
            # table.setItem(ix.row(),1,checkbox.setChecked(checkboxval))
            row = ix.row()
            properties[row][1] = str(checkboxval)

            # table.setItem[ix.row(),1,"True"]
            # table.setCellWidget(ix.row(), 1, setchecked(checkboxval))

        def onComboIndexChanged(value):
            row = table.currentRow()
            print(row)
            dct = eval(properties[row][1])
            actkey = list(dct.keys())[value]

            for key, val in dct.items():
                dct[key] = False
                if key ==actkey:
                    dct[key] = True

            properties[row][1] = str(dct)

        def onButtonClicked():
            allRows = table.rowCount()
            print("")
            print("start")
            for row in range(0, allRows):
                try:
                    twi0 = table.item(row, 0).text()
                    twi1 = table.item(row, 1).text()
                    properties[row][0] = twi0
                    properties[row][1] = twi1
                except:
                    pass  # skip bool

            print("end")
            print("")
            self.close()

        # Grid Layout
        grid = QtWidgets.QGridLayout()

        self.setLayout(grid)
        self.setWindowTitle(text)
        newfont = QtGui.QFont("Sans Serif", 10)

        # Data
        data = properties

        # Create Empty 5x5 Table
        table = QtWidgets.QTableWidget(self)

        table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents)

        table.setRowCount(len(properties))
        table.setColumnCount(2)
        table.horizontalHeader().setVisible(False)
        table.verticalHeader().setVisible(False)

        # Enter data onto Table
        horHeaders = []
        i = 0
        highlighted_index = 0

        self.setFont(newfont)
        for item in data:
            table.setItem(i, 0, QtWidgets.QTableWidgetItem(item[0]))
            table.item(i, 0).setFlags(QtCore.Qt.ItemIsEditable)

            if (item[1] == 'True' or item[1] == 'False'):
                checkbox = QtWidgets.QCheckBox()
                if item[1] == 'True':
                    checkbox.setChecked(True)

                else:
                    checkbox.setChecked(False)
                table.setCellWidget(i, 1, checkbox)

                # checkbox.isChecked()
                checkbox.clicked.connect(onCheckBoxStateChanged)
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
                    table.setCellWidget(i, 1, combo)
                    combo.setCurrentIndex(highlighted_index)

                    combo.currentIndexChanged.connect(onComboIndexChanged)
                    # combo.currentIndexChanged.connect(onCurrentIndexChanged)

                else:
                    table.setItem(i, 1, QtWidgets.QTableWidgetItem(item[1]))
            i = i + 1

        #print(highlighted_index)

        # Add Header
        table.setHorizontalHeaderLabels(horHeaders)

        # Adjust size of Table, first column adjusted to text, 2nd column adjusted to max 400
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        w0 = table.columnWidth(0)
        w1 = table.columnWidth(1)
        w0=int(w0*1.1)
        w1=int(w1*1.2)
        #print(w0,w1)
        if w0>400:
            w0=400
        if w1>400:
            w1=400
        table.setColumnWidth(0, w0)
        table.setColumnWidth(1, w1)

        # Add Table to Grid
        grid.addWidget(table, 0, 0)

        lines = len(data)
        if (lines > 25):
            lines = 25
        newheight = 100 + lines * 28

        Qinfo = QtWidgets.QLabel(info)

        okButton = QtWidgets.QPushButton("OK")
        if len(info) != 0:
            grid.addWidget(Qinfo)
        grid.addWidget(okButton)
        #self.setGeometry(200, 200, 600, newheight)

        #table.setSizeAdjustPolicy(
        #    QtWidgets.QAbstractScrollArea.AdjustToContents)

        table.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)


        self.show()
        self.raise_()

        okButton.clicked.connect(onButtonClicked)

        return properties


class MultiButtonWidget(QtWidgets.QWidget):

    def __init__(self, listbuttons, name, info):
        QtWidgets.QWidget.__init__(self)
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

        # Qinfo = QtWidgets.QLineEdit(info)
        # Qinfo.setReadOnly(True)
        Qinfo = QtWidgets.QLabel(info)
        if len(info) != 0:
            layout.addWidget(Qinfo)

    def handleButton(self, data="\n"):
        # print (data)
        self.buttonpressed = data
        # sys.exit(0) #sysexit is slow, while close is fast
        self.close()


class CheckListWidget(QtWidgets.QDialog):

    def __init__(
            self,
            name,
            datalist,
            info
    ):
        QtWidgets.QListWidget.__init__(self)

        self.name = name

        self.model = QtGui.QStandardItemModel()
        self.listView = QtWidgets.QListView()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # stop qtimer assertion when closing

        for string, datatype in datalist:

            item = QtGui.QStandardItem(string)
            item.setCheckable(True)
            check = datatype
            # print(string,check)
            if (check == True):
                item.setCheckState(2)
            else:
                item.setCheckState(0)
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
        '''
        self.choices = [self.model.item(i).text() for i in
                        range(self.model.rowCount())
                        if self.model.item(i).checkState()
                        == QtCore.Qt.Checked]
        '''
        itemlist = []
        for i in range(self.model.rowCount()):
            if self.model.item(i).checkState() == 2:
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

    def __init__(
            self,
            name,
            datalist,
            info
    ):
        QtWidgets.QListWidget.__init__(self)

        self.name = name

        # self.model = QtGui.QSta.QStandardItemModel()
        self.vbox = QtWidgets.QVBoxLayout()

        # self.listView = QtWidgets.QListView()
        self.button_group = QtWidgets.QButtonGroup()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # stop qtimer assertion when closing
        vbox = QtWidgets.QVBoxLayout(self)

        i = 0

        # check state of all buttons, prevent that all buttons are disabled, if multiple buttons are selected last one is enabled.
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
        '''
        self.choices = [self.model.item(i).text() for i in
                        range(self.model.rowCount())
                        if self.model.item(i).checkState()
                        == QtCore.Qt.Checked]
        '''

        self.choices = self.button_group.checkedId()
        self.accept()

    def onAbort(self):
        self.choices = None
        self.reject()


class ComboBoxWidget(QtWidgets.QDialog):

    def __init__(
            self,
            name,
            datalist,
            info
    ):
        QtWidgets.QListWidget.__init__(self)

        self.name = name

        # self.model = QtGui.QSta.QStandardItemModel()
        self.vbox = QtWidgets.QVBoxLayout()

        # self.listView = QtWidgets.QListView()
        self.button_group = QtWidgets.QComboBox()

        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # stop qtimer assertion when closing
        vbox = QtWidgets.QVBoxLayout(self)

        i = 0
        highlighted_index=0
        # check state of all buttons, prevent that all buttons are disabled, if multiple buttons are selected last one is enabled.
        anybuttonselected = False
        for string, datatype in datalist:
            if datatype == True:
                anybuttonselected = True
                highlighted_index = i
            i = i + 1
        if anybuttonselected == False:
            datalist[0][1] = True

        i=0
        for string, datatype in datalist:
            self.button_group.addItem(string)
            # self.button_group.setId(self.button_name, i)
            # self.button_name.setChecked(datatype)
            #vbox.addWidget(self.button_group)

            i = i + 1
        vbox.addWidget(self.button_group)

        #highlighted_index=2
        #print('index', highlighted_index)
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


if __name__ == "__main__":
    app = QApplication([])
    w = MainWidget()
    w.show()
    app.exec_()


class ImageListViewWidget(QtWidgets.QListWidget):
    dropped = QtCore.pyqtSignal(list)

    def __init__(self, type, parent=None):
        super(ImageListViewWidget, self).__init__(parent)
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


class ImageListViewForm(QtWidgets.QMainWindow):

    def __init__(self, parent=None, title="Drag and drop image dialog"):
        super(ImageListViewForm, self).__init__(parent)
        self.setWindowTitle(title)
        self.view = ImageListViewWidget(self)

        self.view.dropped.connect(self.pictureDropped)

        # self.connect(self.view, QtCore.SIGNAL("dropped"), self.pictureDropped)
        self.setCentralWidget(self.view)
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

    # DIALOGS QT
    @staticmethod
    def dialog_ok_cancel(text, windowtext="Confirm", alwaysontop=True):
        """Dialog ask Ok or Cancel

        :Parameters: title, text
        :Returns: result
        """

        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

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
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        datalist = []
        for key, val in properties.items():
            datalist.append([key, val])

        window = CheckListWidget(name=windowtext, datalist=datalist, info=info)

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()

        app.exec()
        try:
            table = window.choices
            print("Selection made.")
        except:
            # table = properties
            table = None
            print("Cancel")

        ret = table
        return ret

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
        index=0
        valfinal = 0
        for key, value in dct.items():
            if value == True:
                valfinal = index
            index=index+1
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
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        datalist = []
        for key, val in properties.items():
            datalist.append([key, val])

        window = ComboBoxWidget(name=windowtext, datalist=datalist, info=info)

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()

        app.exec()
        try:
            table = window.choices
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
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        datalist = []
        for key, val in properties.items():
            datalist.append([key, val])

        window = RadioButtonListWidget(name=windowtext, datalist=datalist, info=info)

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()

        app.exec()
        try:
            table = window.choices
            print("Selection made.")
        except:
            # table = properties
            table = None
            print("Cancel")

        ret = table
        return ret

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

        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        window = MultiButtonWidget(listbuttons, windowtext, info)
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()
        app.exec_()
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

        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

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
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.dialog_message(text)

    @staticmethod
    def textbox(text, windowtext="Textbox", alwaysontop=True):
        """Text message
        This is a multi-line textbox
        :Parameters: text
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.dialog_textbox(windowtext, text)
        window.setWindowTitle(windowtext)
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()
        app.exec()
        # sys.exit(app.exec_())

    @staticmethod
    def textbox_html(text, windowtext="Textbox", alwaysontop=True):
        """Text message
        This is a multi-line textbox, input is a HTML string
        :Parameters: text
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        window.dialog_textbox_html(windowtext, text)
        window.setWindowTitle(windowtext)
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()
        app.exec()
        # sys.exit(app.exec_())

    @staticmethod
    def error(text, alwaysontop=True):
        """Error message

        :Parameters: text
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.dialog_error(text)

    @staticmethod
    def dialog_propertygrid(properties, windowtext='Properties', verbose=True, info="", alwaysontop=True):
        """Simple property grid

        :Parameters: property_dictionary,text
        :Returns: property_dictionary

        .. code-block:: python

            properties = {'rows': 3, 'cols':3}
            propertiesafter = em.Dialogs.dialog_propertygrid(properties)
            print(properties)
            print(propertiesafter)

        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        data = []
        datatypes = []

        #avoid triggering errors due to empty strings
        for k, v in properties.items():
            if v == "":
                properties.update({k: " "})

        for key, val in properties.items():
            data.append([key, str(val)])
            datatypes.append(type(val))

        table = window.dialog_propertygrid(data, windowtext, info)

        app.exec()

        # print('out', table)

        properties = {}
        st1 = ''
        for row in range(0, len(table)):
            # try:
            twi0 = table[row][0]
            twi1 = table[row][1]
            st1 = st1 + ' ' + str(twi0) + ' ' + str(twi1)

            if (datatypes[row] is bool):
                res = False
                if (twi1 == 'True'):
                    res = True
                properties[twi0] = res
                # properties[twi0] = data[row][0]
            else:
                if (datatypes[row] is int):
                    try:
                        properties[twi0] = int(twi1)
                    except:
                        properties[twi0] = float(twi1)
                        # changed from int to float
                else:
                    if (datatypes[row] is float):
                        properties[twi0] = float(twi1)
                    else:
                        if (datatypes[row] is dict):
                            properties[twi0] = eval(twi1)
                            print('eval:' , eval(twi1))
                        else:
                            properties[twi0] = twi1  # not float or int therefore making it string

        if (verbose == True):
            print('types: ', st1)
            print('out: ', properties)
        return properties

    @staticmethod
    def openfile_dialog(path='/', windowtext='Open File Dialog',filter="Images (*.png *.jpg *.bmp, *.tif, *.tiff)", alwaysontop=True):
        """
        Open file dialog

        :Parameters: path, text, filter, alwaysontop
        :Returns: path
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(window, windowtext, path,filter)
        return filename

    @staticmethod
    def savefile_dialog(path='/', windowtext='Save File Dialog',filter="Images (*.png *.jpg *.bmp, *.tif, *.tiff)", alwaysontop=True):
        """
        Save file dialog

        :Parameters: path, text, filter, alwaysontop
        :Returns: path
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        filename, _filter = QtWidgets.QFileDialog.getSaveFileName(window, windowtext, path, filter)
        return filename


    @staticmethod
    def openfolder_dialog(path='/', windowtext='Open File Dialog', alwaysontop=True):
        """
        Open folder dialog

        :Parameters: path, text
        :Returns: path
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = Window()
        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                break

        # close all open windows
        cv.destroyAllWindows()

        try:
            pnt = pntslist[0][0]
        except:
            pnt = (0, 0)
        print("point: {0}".format(pntslist))
        return pnt

    """
    @staticmethod
    def _Convert.toRGB(img):
        img1 = img
        channels = len(img.shape)
        if (channels is not 3):
            img1 = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        return img1
    """

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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
        img0 = ims.Image.Convert.toRGB(img0)

        pntslist = []

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            global refPt, cropping

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            if event == cv.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]
                cropping = True

            # check to see if the left mouse button was released
            elif event == cv.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))
                cropping = False

                # draw a line from point 0 to 1
                cv.line(img0, refPt[0], refPt[1], (0, 255, 0), 2)

                refPt2 = [(int(refPt[0][0] * 1 / fx), int(refPt[0][1] * 1 / fx)),
                          (int(refPt[1][0] * 1 / fx), int(refPt[1][1] * 1 / fx))]
                pntslist.append(refPt2)
                cv.imshow(windowtext, img0)

        clone = img0.copy()
        # cv.namedWindow(windowtext)
        cv.setMouseCallback(windowtext, click_and_crop)

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow(windowtext, img0)
            key = cv.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img0 = clone.copy()
            # monitor escape
            elif key == 27:
                break
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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

        '''
        fx = 1024 / img.shape[1]
        if fx < 1:
            frame = cv.resize(img, None, fx=fx, fy=fx)
            img0 = frame.copy()  # avoid drawing inside the copy
        else:
            img0 = img.copy()
        '''

        img0 = img.copy()
        zoomfactor = 1
        fx = zoomfactor

        # print(fx)
        cv.namedWindow(windowtext, cv.WINDOW_AUTOSIZE)

        refPt = []
        cropping = False
        img0 = ims.Image.Convert.toRGB(img0)

        pntslist = []

        def click_and_crop(event, x, y, flags, param):
            # grab references to the global variables
            global refPt, cropping

            # if the left mouse button was clicked, record the starting
            # (x, y) coordinates and indicate that cropping is being
            # performed
            if event == cv.EVENT_LBUTTONDOWN:
                refPt = [(x, y)]
                cropping = True

            # check to see if the left mouse button was released
            elif event == cv.EVENT_LBUTTONUP:
                # record the ending (x, y) coordinates and indicate that
                # the cropping operation is finished
                refPt.append((x, y))
                cropping = False

                # draw a rectangle around the region of interest
                cv.rectangle(img0, refPt[0], refPt[1], (0, 255, 0), 2)
                # print(refPt)
                refPt2 = [(int(refPt[0][0] * 1 / fx), int(refPt[0][1] * 1 / fx)),
                          (int(refPt[1][0] * 1 / fx), int(refPt[1][1] * 1 / fx))]
                # print(refPt2)
                pntslist.append(refPt2)
                cv.imshow(windowtext, img0)

        clone = img0.copy()

        # cv.namedWindow(windowtext)
        cv.setMouseCallback(windowtext, click_and_crop)
        zoomfactor = 1

        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress
            cv.imshow(windowtext, img0)
            key = cv.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                img0 = clone.copy()
            # monitor escape
            elif key == 27:
                break
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                break

            if key == 43:
                zoomfactor = zoomfactor * 1.25
                img0 = cv.resize(img, None, fx=zoomfactor, fy=zoomfactor)
                fx = zoomfactor
            if key == 45:
                zoomfactor = zoomfactor * 0.75
                img0 = cv.resize(img, None, fx=zoomfactor, fy=zoomfactor)
                fx = zoomfactor
            if key == 114:
                zoomfactor = 1
                img0 = cv.resize(img, None, fx=zoomfactor, fy=zoomfactor)
                fx = zoomfactor

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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                break

        cv.destroyAllWindows()
        print("Mask: thresholdedimage, Min={},Max={},Blur={}".format(min, max, blur))
        thresh1 = subfunction(img, min, max, blur)
        return thresh1, min, max, blur

    @staticmethod
    def adjust_mask_with_background(img, windowtext="Select Mask"):
        """create an image mask by setting the intensity range and blur. Returns: image, min,max,blur
        zoom in/out with +/-, reset zoom with r
        hide/unhide mask with h
        :Parameters: image, windowtext=name of form
        :Returns: ImageOut, Min,Max,Blur
        """

        def subfunction(img, min, max, blur, zoomfactor, hidemask):
            img = cv.resize(img, None, fx=zoomfactor, fy=zoomfactor)
            imout = cv.GaussianBlur(img, (blur, blur), 0)
            thresh0 = cv.inRange(imout, min, max)
            thresh1 = ims.Image.Process.Falsecolor.falsecolor_merge2channels(thresh0, imout)
            if hidemask == True:
                thresh1 = img

            return thresh1

        def nothing(x):
            pass

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

        while (1):
            min = cv.getTrackbarPos("Min", windowtext)
            max = cv.getTrackbarPos("Max", windowtext)
            blur = cv.getTrackbarPos("Blur", windowtext)
            if (blur % 2 == 0):
                blur = blur + 1
            thresh1 = subfunction(frame, min, max, blur, zoomfactor, hidemask)

            cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
        thresh1 = subfunction(img, min, max, blur, zoomfactor=1, hidemask=False)
        return thresh1, min, max, blur

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
            min = cv.getTrackbarPos("Min", windowtext)
            blur = cv.getTrackbarPos("Blur", windowtext)
            if (blur % 2 == 0):
                blur = blur + 1
            thresh1, angle = ims.Image.Process.gradient_image_nonmaxsuppressed(frame, blur, min)

            cv.imshow(windowtext, thresh1)
            k = cv.waitKey(1) & 0xFF
            if k == 27:
                break
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                break

        cv.destroyAllWindows()

        mask = cv.morphologyEx(img, cv.MORPH_CLOSE, se1)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, se2)
        mask = cv.morphologyEx(mask, cv.MORPH_ERODE, se3)
        thresh1 = cv.morphologyEx(mask, cv.MORPH_DILATE, se4)
        print("OpenCloseErodeDilate: Image")
        return thresh1

    '''
    @staticmethod
    def adjust_HSV(img, windowtext="HSV Channels"):
        """Adjust HSV channels dialog - Dialog which enables adjustment of hsv channels (hue, saturation and value).

        :Parameters: windowtext=name of form
        :Returns: image, h,s,v
        """

        def subfunction(img2,h,s,v):
            # Normal masking algorithm
            lower_blue = np.array([h, s, v])
            upper_blue = np.array([180, 255, 255])

            mask = cv.inRange(img2, lower_blue, upper_blue)

            result = cv.bitwise_and(frame, frame, mask=mask)
            return result

        def nothing(x):
            pass


        fx= 1024/img.shape[1]
        frame = cv.resize(img,None,fx=fx,fy=fx)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Creating a window for later use
        #cv.namedWindow(windowtext)
        cv.namedWindow(windowtext)

        # Starting with 100's to prevent error while masking
        h, s, v = 100, 100, 100

        # Creating track bar
        cv.createTrackbar('Hue', windowtext, 0, 179, nothing)
        cv.createTrackbar('Saturation', windowtext, 0, 255, nothing)
        cv.createTrackbar('Value', windowtext, 0, 255, nothing)


        while (1):

            # converting to HSV

            # get info from track bar and appy to result
            h = cv.getTrackbarPos('Hue', windowtext)
            s = cv.getTrackbarPos('Saturation', windowtext)
            v = cv.getTrackbarPos('Value', windowtext)

            
            result = subfunction(hsv,h,s,v)

            cv.imshow(windowtext, result)

            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                break
        cv.destroyAllWindows()
        print("output: image, Hue={},Sat={},Val={}".format(h,s,v))
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        result = subfunction(hsv, h, s, v)
        return result, h,s,v
    '''

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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
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
            if cv.getWindowProperty(windowtext, cv.WND_PROP_AUTOSIZE) < 1:
                break

        cv.destroyAllWindows()

    @staticmethod
    def dialog_imagelistview(windowtext="Drag and drop image dialog", alwaysontop=True):
        """
        Image listview dialog. The listview allows for dragging dropping and thumbnail previewing of files.

        :Returns: list of urls
        """
        #workaround to avoid multiple instances of QtWidget causing memory leaks, required for QT5.14.2, better method available?
        app = QtCore.QCoreApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        window = ImageListViewForm()

        if alwaysontop == True:
            window.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        window.show()
        app.exec()
        url_list = window.url_list
        return url_list

# main
