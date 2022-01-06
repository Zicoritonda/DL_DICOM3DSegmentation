# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:49:07 2020

@author: User
"""
import skimage
import skimage.feature
import skimage.viewer
from skimage.viewer.widgets import Slider, Button
from skimage.viewer.qt import QtWidgets, QtGui, Qt, Signal
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from tkinter import *
from skimage.viewer.plugins.overlayplugin import OverlayPlugin
import matplotlib.colors as mcolors
from pandas import *
# from skimage.viewer.plugins.canny import CannyPlugin
# from skimage.viewer.canvastools import PaintTool
from skimage.viewer.canvastools.base import CanvasToolBase, ToolHandles
from matplotlib import lines
import matplotlib.colors as mcolors
from matplotlib.widgets import RectangleSelector
# LABELS_CMAP = mcolors.ListedColormap(mcolors.CSS4_COLORS)
LABELS_CMAP = mcolors.ListedColormap(['white', 'red', 'dodgerblue', 'gold',
                                      'greenyellow', 'blueviolet'])
from matplotlib import backend_bases
# mpl.rcParams['toolbar'] = 'None'
backend_bases.NavigationToolbar2.toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Back', 'Back to  previous view', 'back', 'back'),
        ('Forward', 'Forward to next view', 'forward', 'forward'),
        (None, None, None, None),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
        (None, None, None, None),
        ('Save', 'Save the figure', 'filesave', 'save_figure'),
      )

from skimage.util.dtype import dtype_range
from skimage.exposure import rescale_intensity
from skimage.viewer.qt import QtWidgets, QtGui, Qt, Signal
from skimage.viewer.utils import (dialogs, init_qtapp, figimage, start_qtapp,
                     update_axes_image)
from skimage.viewer.utils.canvas import BlitManager, EventManager
from skimage.viewer.plugins.base import Plugin

arraysegmented = []
slices = 0
pixelarray = []
labels = 1
firstimage = []
pixelvalue = [0, 50, 100, 150, 200, 250]
#otak,otak kecil,corpus,batang otak,ventrikel

filename = "tes/predict.bin"#"data/annotated/Train/Ahmad Aris_annotated6.bin"
outfile = "output/13.bin"#"data/annotated/Train/Ahmad Aris_annotated7.bin"
f = open(filename, "rb")
# np.frombuffer(f.read(52), dtype='f')
ndata = np.frombuffer(f.read(), dtype=np.uint8).reshape(256, 256, 256)
data = ndata.copy()
data[data!=0] = 255
#data2 = np.frombuffer(open(outfile, 'rb').read(), dtype=np.uint8).reshape(256, 256, 256)
new_data = np.zeros((256, 256, 256), dtype='uint8')
# Mask Sagital ke array baru
for i in range(256):
    new_data[:,:,i] = data[:,:,i]
# Mask Coronal ke array baru
for i in range(256):
    new_data[:,i,:] = data[:,i,:]
# Mask Axial ke array baru
for i in range(256):
    new_data[i,:,:] = data[i,:,:]

#new_data[data2==150] = 150
#new_data[119:136] = data[119:136]
class PaintTool(CanvasToolBase):
    def __init__(self, manager, overlay_shape, radius=1, alpha=0.2,
                 on_move=None, on_release=None, on_enter=None,
                 rect_props=None):
        super(PaintTool, self).__init__(manager, on_move=on_move,
                                        on_enter=on_enter,
                                        on_release=on_release)

        props = dict(edgecolor='r', facecolor='0.7', alpha=0.5, animated=True)
        props.update(rect_props if rect_props is not None else {})

        self.alpha = alpha
        self.cmap = LABELS_CMAP
        self._overlay_plot = None
        self.shape = overlay_shape[:2]

        self._cursor = plt.Rectangle((0, 0), 0, 0, **props)
        self._cursor.set_visible(False)
        self.ax.add_patch(self._cursor)

        # `label` and `radius` can only be set after initializing `_cursor`
        self.label = labels
        self.radius = radius

        # Note that the order is important: Redraw cursor *after* overlay
        self.artists = [self._overlay_plot, self._cursor]
        self.manager.add_tool(self)

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        if value >= self.cmap.N:
            raise ValueError('Maximum label value = %s' % len(self.cmap - 1))
        self._label = value
        
        self._cursor.set_edgecolor(self.cmap(value))

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, r):
        self._radius = r
        self._width = 2 * r + 1
        self._cursor.set_width(self._width)
        self._cursor.set_height(self._width)
        self.window = CenteredWindow(r, self._shape)

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        global firstimage
        firstimage = image
        self._overlay = image
        # print(self._overlay_plot)
        if image is None:
            self.ax.images.remove(self._overlay_plot)
            self._overlay_plot = None
        elif self._overlay_plot is None:
            props = dict(cmap=self.cmap, alpha=self.alpha,
                          norm=mcolors.NoNorm(vmin=0, vmax=self.cmap.N),
                          animated=True)
            self._overlay_plot = self.ax.imshow(image, **props)
        else:
            self._overlay_plot.set_data(image)
        
        self.redraw()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        if not self._overlay_plot is None:
            self._overlay_plot.set_extent((-0.1, shape[1] + 0.1,
                                           shape[0] + 0.1, -0.1))
            self.radius = self._radius
        self.overlay = np.zeros(shape, dtype='uint8')

    def on_key_press(self, event):
        global labels
        if event.key == 'enter':
            if labels==5:
                self.overlay = np.zeros(self.shape, dtype='uint8')
                labels = 0
                self.label = labels
                
            self.callback_on_enter(self.geometry)
            self.redraw()
            # self.remove_overlay(event)
            oldlabel = self.label
            self.label = oldlabel+1
            labels = self.label
            print(labels)
            
    def on_mouse_press(self, event):
        if event.button not in [1, 3] or not self.ax.in_axes(event):
            return
        self.update_cursor(event.xdata, event.ydata)
        self.update_overlay(event.xdata, event.ydata, event.button)
        # self.get_coordinat(event.xdata, event.ydata)

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        self.callback_on_release(self.geometry)

    def on_move(self, event):
        if not self.ax.in_axes(event):
            self._cursor.set_visible(False)
            self.redraw() # make sure cursor is not visible
            return
        self._cursor.set_visible(True)
        self.update_cursor(event.xdata, event.ydata)
        # self.on_mouse_press(event)
        
        if event.button not in [1, 3]:
            self.redraw() # update cursor position
            return
        self.update_overlay(event.xdata, event.ydata, event.button)
        self.callback_on_move(self.geometry)

    def update_overlay(self, x, y, button):
        overlay = self.overlay
        if button == 1:
            overlay[self.window.at(y, x, button)] = self.label
        else:
            overlay[self.window.at(y, x, button)] = 0
        # Note that overlay calls `redraw`
        self.overlay = overlay

    def update_cursor(self, x, y):
        x = x - self.radius - 1
        y = y - self.radius - 1
        self._cursor.set_xy((x, y))

    def get_coordinat(self, x, y):
        # print(int(round(x, 0)), int(round(y, 0)), self.label)
        print("hey")
        arraysegmented.append([slices, int(round(x, 0)), int(round(y, 0)), self.label])
        
    def remove_overlay(self, image):
        self.overlay()
        

    def on_scroll(self, event):
        if event.button == 'up':
            r = self._radius
            r = r + 1 if r + 1 < 5 else 5
            self.radius = r
            print('going up bcs im not feeling down')
        else:
            r = self._radius
            r = r - 1 if r - 1 > 1 else 1
            self.radius = r
            print('going down bcs nothing is up')
            
        self.redraw()
        
    @property
    def geometry(self):
        return self.overlay

class CenteredWindow(object):
    """Window that create slices numpy arrays over 2D windows.
    Examples
    --------
    >>> a = np.arange(16).reshape(4, 4)
    >>> w = CenteredWindow(1, a.shape)
    >>> a[w.at(1, 1)]
    array([[ 0,  1,  2],
           [ 4,  5,  6],
           [ 8,  9, 10]])
    >>> a[w.at(0, 0)]
    array([[0, 1],
           [4, 5]])
    >>> a[w.at(4, 3)]
    array([[14, 15]])
    """
    def __init__(self, radius, array_shape):
        self.radius = radius
        self.array_shape = array_shape

    def at(self, row, col, button):
        h, w = self.array_shape
        r = round(self.radius)
        # Note: the int() cast is necessary because row and col are np.float64,
        # which does not get cast by round(), unlike a normal Python float:
        # >>> round(4.5)
        # 4
        # >>> round(np.float64(4.5))
        # 4.0
        # >>> int(round(np.float64(4.5)))
        # 4
        row, col = int(round(row)), int(round(col))
        xmin = max(0, col - r)
        xmax = min(w, col + r + 1)
        ymin = max(0, row - r)
        ymax = min(h, row + r + 1)
        print(ymin, ymax, xmin, xmax)
        self.radiusPixel(ymin, ymax, xmin, xmax, button)
        
        return (slice(ymin, ymax), slice(xmin, xmax))
    
    def radiusPixel(self, ymin, ymax, xmin, xmax, button):
        for i in range(ymin, ymax):
            for j in range(xmin, xmax):
                if button == 1:
                    arraysegmented.append([slices, int(round(j, 0)), int(round(i, 0)), labels])
                else:
                    arraysegmented.append([slices, int(round(j, 0)), int(round(i, 0)), 0])
                # print(arraysegmented)
                
                
class ImageViewer(QtWidgets.QMainWindow):
    dock_areas = {'top': Qt.TopDockWidgetArea,
                  'bottom': Qt.BottomDockWidgetArea,
                  'left': Qt.LeftDockWidgetArea,
                  'right': Qt.RightDockWidgetArea}

    # Signal that the original image has been changed
    original_image_changed = Signal(np.ndarray)

    def __init__(self, image, useblit=True):
        # Start main loop
        init_qtapp()
        super(ImageViewer, self).__init__()

        # TODO: Add ImageViewer to skimage.io window manager

        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Image Viewer")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('Open file', self.open_file,
                                 Qt.CTRL + Qt.Key_O)
        self.file_menu.addAction('Save to file', self.save_to_file,
                                 Qt.CTRL + Qt.Key_S)
        self.file_menu.addAction('Quit', self.close,
                                 Qt.CTRL + Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        if isinstance(image, Plugin):
            plugin = image
            image = plugin.filtered_image
            plugin.image_changed.connect(self._update_original_image)
            # When plugin is started, start
            plugin._started.connect(self._show)

        self.fig, self.ax = figimage(image)
        self.canvas = self.fig.canvas
        self.canvas.setParent(self)
        self.ax.autoscale(enable=False)

        self._tools = []
        self.useblit = useblit
        if useblit:
            self._blit_manager = BlitManager(self.ax)
        self._event_manager = EventManager(self.ax)

        self._image_plot = self.ax.images[0]
        self._update_original_image(image)
        self.plugins = []

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.canvas)

        status_bar = self.statusBar()
        self.status_message = status_bar.showMessage
        sb_size = status_bar.sizeHint()
        cs_size = self.canvas.sizeHint()
        self.resize(cs_size.width(), cs_size.height() + sb_size.height())

        self.connect_event('motion_notify_event', self._update_status_bar)

    def __add__(self, plugin):
        """Add plugin to ImageViewer"""
        plugin.attach(self)
        self.original_image_changed.connect(plugin._update_original_image)

        if plugin.dock:
            location = self.dock_areas[plugin.dock]
            dock_location = Qt.DockWidgetArea(location)
            dock = QtWidgets.QDockWidget()
            dock.setWidget(plugin)
            dock.setWindowTitle(plugin.name)
            self.addDockWidget(dock_location, dock)

            horiz = (self.dock_areas['left'], self.dock_areas['right'])
            dimension = 'width' if location in horiz else 'height'
            self._add_widget_size(plugin, dimension=dimension)

        return self

    def _add_widget_size(self, widget, dimension='width'):
        widget_size = widget.sizeHint()
        viewer_size = self.frameGeometry()

        dx = dy = 0
        if dimension == 'width':
            dx = widget_size.width()
        elif dimension == 'height':
            dy = widget_size.height()

        w = viewer_size.width()
        h = viewer_size.height()
        self.resize(w + dx, h + dy)

    def open_file(self, filename=None):
        """Open image file and display in viewer."""
        if filename is None:
            filename = dialogs.open_file_dialog()
        if filename is None:
            return
        image = io.imread(filename)
        self._update_original_image(image)

    def update_image(self, image):
        """Update displayed image.
        This method can be overridden or extended in subclasses and plugins to
        react to image changes.
        """
        self._update_original_image(image)

    def _update_original_image(self, image):
        self.original_image = image     # update saved image
        self.image = image.copy()       # update displayed image
        self.original_image_changed.emit(image)

    def save_to_file(self, filename=None):
        """Save current image to file.
        The current behavior is not ideal: It saves the image displayed on
        screen, so all images will be converted to RGB, and the image size is
        not preserved (resizing the viewer window will alter the size of the
        saved image).
        """
        if filename is None:
            filename = dialogs.save_file_dialog()
        if filename is None:
            return
        if len(self.ax.images) == 1:
            io.imsave(filename, self.image)
        else:
            underlay = mpl_image_to_rgba(self.ax.images[0])
            overlay = mpl_image_to_rgba(self.ax.images[1])
            alpha = overlay[:, :, 3]

            # alpha can be set by channel of array or by a scalar value.
            # Prefer the alpha channel, but fall back to scalar value.
            if np.all(alpha == 1):
                alpha = np.ones_like(alpha) * self.ax.images[1].get_alpha()

            alpha = alpha[:, :, np.newaxis]
            composite = (overlay[:, :, :3] * alpha +
                         underlay[:, :, :3] * (1 - alpha))
            io.imsave(filename, composite)

    def closeEvent(self, event):
        self.close()

    def _show(self, x=0):
        self.move(x, 0)
        for p in self.plugins:
            p.show()
        super(ImageViewer, self).show()
        self.activateWindow()
        self.raise_()

    def show(self, main_window=True):
        """Show ImageViewer and attached plugins.
        This behaves much like `matplotlib.pyplot.show` and `QWidget.show`.
        """
        self._show()
        if main_window:
            start_qtapp()
        return [p.output() for p in self.plugins]

    def redraw(self):
        if self.useblit:
            self._blit_manager.redraw()
        else:
            self.canvas.draw_idle()

    @property
    def image(self):
        return self._img

    @image.setter
    def image(self, image):
        self._img = image
        update_axes_image(self._image_plot, image)

        # update display (otherwise image doesn't fill the canvas)
        h, w = image.shape[:2]
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)

        # update color range
        clim = dtype_range[image.dtype.type]
        if clim[0] < 0 and image.min() >= 0:
            clim = (0, clim[1])
        self._image_plot.set_clim(clim)

        if self.useblit:
            self._blit_manager.background = None

        self.redraw()

    def reset_image(self):
        self.image = self.original_image.copy()

    def connect_event(self, event, callback):
        """Connect callback function to matplotlib event and return id."""
        cid = self.canvas.mpl_connect(event, callback)
        return cid

    def disconnect_event(self, callback_id):
        """Disconnect callback by its id (returned by `connect_event`)."""
        self.canvas.mpl_disconnect(callback_id)

    def _update_status_bar(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            self.status_message(self._format_coord(event.xdata, event.ydata))
        else:
            self.status_message('')

    def add_tool(self, tool):
        if self.useblit:
            self._blit_manager.add_artists(tool.artists)
        self._tools.append(tool)
        self._event_manager.attach(tool)

    def remove_tool(self, tool):
        if tool not in self._tools:
            return
        if self.useblit:
            self._blit_manager.remove_artists(tool.artists)
        self._tools.remove(tool)
        self._event_manager.detach(tool)

    def _format_coord(self, x, y):
        # callback function to format coordinate display in status bar
        x = int(x + 0.5)
        y = int(y + 0.5)
        try:
            return "%4s @ [%4s, %4s]" % (self.image[y, x], x, y)
        except IndexError:
            return ""
        
        
class CollectionViewer(ImageViewer):
    def __init__(self, image_collection, update_on='move', **kwargs):
        self.image_collection = image_collection
        self.index = 0
        self.num_images = len(self.image_collection)

        first_image = image_collection[0]
        super(CollectionViewer, self).__init__(first_image)

        slider_kws = dict(value=0, low=0, high=self.num_images - 1)
        slider_kws['update_on'] = update_on
        slider_kws['callback'] = self.update_index
        slider_kws['value_type'] = 'int'
        self.slider = Slider('frame', **slider_kws)
        self.layout.addWidget(self.slider)

        # TODO: Adjust height to accommodate slider; the following doesn't work
        # s_size = self.slider.sizeHint()
        # cs_size = self.canvas.sizeHint()
        # self.resize(cs_size.width(), cs_size.height() + s_size.height())

    def update_index(self, name, index):
        """Select image on display using index into image collection."""
        index = int(round(index))
        global slices
        slices = index

        if index == self.index:
            return

        # clip index value to collection limits
        index = max(index, 0)
        index = min(index, self.num_images - 1)

        self.index = index
        self.slider.val = index
        self.update_image(self.image_collection[index])

    def keyPressEvent(self, event):
        if type(event) == QtGui.QKeyEvent:
            key = event.key()
            # Number keys (code: 0 = key 48, 9 = key 57) move to deciles
            if 48 <= key < 58:
                index = int(0.1 * (key - 48) * self.num_images)
                self.update_index('', index)
                event.accept()
            else:
                event.ignore()
        else:
            event.ignore()


class main:
    def __init__(self,  window):
        global data
        self.data = data
        self.imglist = []
        self._job = None
        self.window = window
        window.title("Manual Brain Segmentation")
        window.geometry("520x180")
        self.slidervalue = 0
        buttonCoronal = Button (window, text="Sagital", command=self.sagital, height = 2, width=20)
        buttonSagital = Button (window, text="Coronal", command=self.coronal, height = 2, width=20)
        buttonAxial = Button (window, text="Axial", command=self.axial, height = 2, width=20)
        buttonReplaceCoronal = Button (window, text="Replace Pixel Array Sagital", command=self.replacePixelSagital, height = 2, width=20)
        buttonReplaceSagital = Button (window, text="Replace Pixel Array Coronal", command=self.replacePixelCoronal, height = 2, width=20)
        buttonReplaceAxial = Button (window, text="Replace Pixel Array Axial", command=self.replacePixelAxial, height = 2, width=20)
        buttonGenerateBin = Button (window, text="Generate Segmentation to Bin", command=self.generateBin, height = 2, width=70)
        
        # GRID
        buttonCoronal.grid(row=0, column=0, pady=10, padx=10)
        buttonSagital.grid(row=0, column=1, pady=10, padx=10)
        buttonAxial.grid(row=0, column=2, pady=10, padx=10)
        buttonReplaceCoronal.grid(row=1, column=0, pady=10, padx=10)
        buttonReplaceSagital.grid(row=1, column=1, pady=10, padx=10)
        buttonReplaceAxial.grid(row=1, column=2, pady=10, padx=10)
        buttonGenerateBin.grid(row=2, column=0, pady=10, padx=10, columnspan=3)
        
    def viewSkimage(self):
        viewer = skimage.viewer.CollectionViewer(self.imglist)
        viewer.show()
        
    def sagital(self):
        global labels
        labels = 1
        self._job = None
        self.imglist.clear()
        for i in range(256):
            self.imglist.append(self.data[:,:,i])
        
        viewer = CollectionViewer(self.imglist)
        PaintTool(viewer, self.data.shape)
        viewer.show()
        
    def coronal(self):
        global labels
        labels = 1
        self._job = None
        self.imglist.clear()
        for i in range(256):
            self.imglist.append(self.data[:,i,:])
            
        viewer = CollectionViewer(self.imglist)
        PaintTool(viewer, self.data.shape)
        viewer.show()
        
    def axial(self):
        global labels
        labels = 1
        self._job = None
        self.imglist.clear()
        for i in range(256):
            self.imglist.append(self.data[i,:,:])
            
        viewer = CollectionViewer(self.imglist)
        PaintTool(viewer, self.data.shape)
        viewer.show()

    def replacePixelSagital(self):
        global new_data
        global arraysegmented
        newarray = np.array(arraysegmented)
        arraylabelling = DataFrame(newarray).drop_duplicates().values
        for i in arraylabelling:
            s = i[0]
            x = i[1]
            y = i[2]
            l = i[3]
            getpixel = new_data[y][x][s]
            # if getpixel != 0:            
            new_data[y][x][s] = pixelvalue[l]
            
        arraysegmented.clear()
        print(arraysegmented)
        self._job = None
        self.imglist.clear()
        for i in range(256):
            self.imglist.append(new_data[:,:,i])
            
        viewer = skimage.viewer.CollectionViewer(self.imglist)
        viewer.show()
        
    def replacePixelCoronal(self):
        global new_data
        global arraysegmented
        newarray = np.array(arraysegmented)
        # print(DataFrame(newarray).head())
        arraylabelling = DataFrame(newarray).drop_duplicates(subset=[0, 1, 2], keep='last').values
        for i in arraylabelling:
            s = i[0]
            x = i[1]
            y = i[2]
            l = i[3]
            getpixel = new_data[y][s][x]
            # if getpixel != 0:
            if l != 0:            
                new_data[y][s][x] = pixelvalue[l]
            
        arraysegmented.clear()
        self._job = None
        self.imglist.clear()
        for i in range(256):
            self.imglist.append(new_data[:,i,:])
            
        viewer = skimage.viewer.CollectionViewer(self.imglist)
        viewer.show()

    def replacePixelAxial(self):
        global new_data
        global arraysegmented
        newarray = np.array(arraysegmented)
        arraylabelling = DataFrame(newarray).drop_duplicates().values
    
        for i in arraylabelling:
            s = i[0]
            x = i[1]
            y = i[2]
            l = i[3]
            getpixel = new_data[s][y][x]
            # if getpixel != 0:            
            new_data[s][y][x] = pixelvalue[l]
            
        arraysegmented.clear()
        self._job = None
        self.imglist.clear()
        for i in range(256):
            self.imglist.append(new_data[i,:,:])
            
        viewer = skimage.viewer.CollectionViewer(self.imglist)
        viewer.show()

    def generateBin(self):
        global new_data
        print(new_data.size)
        filebin = open(outfile, "wb")
        filebin.write(new_data)
        filebin.close()
        print("Generate File Successfully")

window= Tk()
start= main (window)
window.mainloop()
