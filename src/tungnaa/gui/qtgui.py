import time
import sys
import random
# import argparse
import queue
import functools as ft
import threading
import numpy as np
import numpy.typing as npt
import typing
from pathlib import Path
from importlib import resources

import fire
from PySide6 import QtCore, QtWidgets, QtGui, QtCharts
import pyqtgraph as pg
import pythonosc
import pythonosc.osc_server

import tungnaa.gui
from tungnaa.gui.downloads import dl_model

# multiprocessing: need to use spawn (no fork on mac, no forkserver on windows)
# note, this may preclude pyinstaller...

# to convert obj.method(*args) to multiprocessing...
import multiprocessing as mp
class Proxy:
    """
    this is a somewhat of a hack to try running the backend in a separate
    process with otherwise minimal changes.

    wrapping the Backend object in a Proxy allows the frontend to asynchronously call methods on the backend (but they don't return)

    all communication from the backend to the frontend is through a Pipe,
    which works basically the same as it was previously with a Queue.

    it's not possible to read attributes of the backend.

    to me this design is suggesting to use asyncio to allow backend calls to
    return asynchronously when needed, and the frontend to use `await` when convenient.

    IPyC (https://pypi.org/project/IPyC/) might be helpful there.

    alternatively, Futures as in concurrent.futures (https://docs.python.org/3/library/concurrent.futures.htm) could be useful -- but I'm unsure whether ProcessPoolExecutor is a good fit for calling methods of one long-lived object in another process.

    However, this pattern might turn out to work fine -- the frontend 
    passes data/triggers to the backend asynchronously, and the backend
    streams state updates to the frontend via the Pipe.
    """
    def __init__(self, obj):
        ctx = mp.get_context('spawn')
        # backend object
        self.obj = obj
        # parent and child process ends of the pipe
        self.parent_conn, child_conn = ctx.Pipe()
        # child process eventually sets this to true and sends it through the pipe
        self.running = False
        # run in a new process
        self.proc = ctx.Process(target=self._run, args=(child_conn,))
        self.proc.start()

    def __getattr__(self, name):
        # print(self.__dict__)
        if name in self.__dict__:
            return self.__dict__[name]
        elif len(self.__dict__):
            if not hasattr(self.obj, name):
                # this is checking the parent process copy of obj, meaning
                # it can only find attributes which exist before the fork
                raise AttributeError
            
            if not self.running:
                # first value received from the backend process says it has started
                # if self.parent_conn.poll():
                try:
                    assert self.parent_conn.recv()
                except AssertionError:
                    print("""backend process failed to start""")
                    raise
                self.running = True
                # else:
                    # return lambda *a, **kw: print(f'skipping {name} (backend not running yet)')
            
            attr = getattr(self.obj, name)
            if callable(attr):
                # return a function which when called, defers onto the backend process
                def f(*a, **kw):
                    # print(name)
                    self.parent_conn.send([name, a, kw])
                return f
            else:
                print("""backend attribute not callable""")
            
            raise NotImplementedError
        else:
            # this case allows unpickling
            raise AttributeError
    
    def _run(self, conn):
        print('======================BACKEND RUN======================')
        self.obj.run(conn)
        self.running = True
        conn.send(self.running)

        while True:
            # call function in backend process asynchronously (no return)
            name, a, kw = conn.recv()
            # print(name)
            getattr(self.obj, name)(*a, **kw)


# save some boilerplate
def Layout(*items, cls=QtWidgets.QHBoxLayout, spacing=None, margins=None):
    l = cls()
    if spacing is not None:
        l.setSpacing(spacing)
    if margins is not None:
        l.setContentsMargins(*margins)
    for item in items:
        if isinstance(item, QtWidgets.QWidget):
            l.addWidget(item)
        elif isinstance(item, QtWidgets.QLayout):
            l.addLayout(item)
    return l
HBoxLayout = ft.partial(Layout, cls=QtWidgets.QHBoxLayout)
VBoxLayout = ft.partial(Layout, cls=QtWidgets.QVBoxLayout)

class DoubleSlider(QtWidgets.QSlider):
    """
    Continuous value QSlider that uses native double values instead of integers
    """

    # define a signal which enables Qt's callback system of signals/slots
    doubleValueChanged = QtCore.Signal(float)

    def __init__(self, decimals=3, *args, **kargs):
        super(DoubleSlider, self).__init__( *args, **kargs)
        self._multi = 10 ** decimals
        self.valueChanged.connect(self.emitDoubleValueChanged)

    def emitDoubleValueChanged(self):
        value = float(super(DoubleSlider, self).value())/self._multi
        self.doubleValueChanged.emit(value)

    def value(self):
        return float(super(DoubleSlider, self).value()) / self._multi

    def setMinimum(self, value):
        return super(DoubleSlider, self).setMinimum(value * self._multi)

    def setMaximum(self, value):
        return super(DoubleSlider, self).setMaximum(value * self._multi)

    def setSingleStep(self, value):
        return super(DoubleSlider, self).setSingleStep(value * self._multi)

    def singleStep(self):
        return float(super(DoubleSlider, self).singleStep()) / self._multi

    def setValue(self, value:float):
        super(DoubleSlider, self).setValue(int(value * self._multi))


class PlotWidget(pg.PlotWidget):
    # disable zoom
    def wheelEvent(self, ev):
        pass

class AlignmentImageItem(pg.ImageItem):
    def __init__(self, clicked):
        super().__init__()
        self.clicked = clicked
    def mousePressEvent(self, ev):
        if self.clicked is not None:
            self.clicked(ev)

class AlignmentGraph(QtWidgets.QWidget):
    """
    Widget for displaying alignments
    """
    def __init__(self, 
        parent:QtWidgets.QWidget=None, 
        image_clicked:typing.Callable=None):
        super().__init__(parent)

        self.max_display_steps = 200

        pg.setConfigOption('imageAxisOrder', 'row-major') # best performance image data must be (height, width)
        #pg.setConfigOption('useNumba', True) # supposedly better performance for image data

        self.num_encodings = 40

        self.frame = 0
        self.prev_tok = 0

        self.plot_widget = pg.GraphicsLayoutWidget(self)
        self.plot_widget.ci.layout.setContentsMargins(0,0,0,4)
        self.plot_widget.ci.layout.setSpacing(0)
        self.plot = self.plot_widget.addPlot()

        # self.plot = PlotWidget(parent=self)
        # See: https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/imageitem.html#pyqtgraph.ImageItem
        self.imageitem = AlignmentImageItem(image_clicked)
        # self.imageitem.setImage(image=self.imagedata.T)
        self.plot.addItem(self.imageitem)
        self.plot.showAxes(True, showValues=(True, True, True, False))
        self.plot.invertY(False) # vertical axis zero at the bottom
        # no idea why, but fully transparent background is too light
        # black with alpha of 40 seems to match
        self.plot_widget.setBackground((0,0,0,40))

        self.attn_slider_resolution = 512
        self.attn_slider_max_value = (
            self.num_encodings-1) * self.attn_slider_resolution
        self.alignment_slider = QtWidgets.QSlider(
            parent=self, orientation=QtCore.Qt.Horizontal)
        self.alignment_slider.setMinimum(0)
        self.alignment_slider.setMaximum( self.attn_slider_max_value )

        self.layout = VBoxLayout(
            self.alignment_slider, self.plot_widget, 
            spacing=0, margins=(0,0,0,0))
        self.setLayout(self.layout)

        self.text = None


    def set_normalized_alignment(self, value:float):
        """
        Sets alignment slider to a normalized position from 0-1

        TODO: There's some kind of bug here when controlling via OSC, where the /set_alignment command doesn't scale properly from 0-1 to the range of tokens
                my guess is that self.attn_slider_max_value isn't getting updated? Or something else weird is going on...
        """
        val_as_token = int(value*self.attn_slider_max_value)
        self.alignment_slider.setValue(val_as_token)
        print(f"Set normalized alignment {value} - as token: {val_as_token}/{self.attn_slider_max_value}/{self.num_encodings} ")

    def set_alignment_as_token_idx(self, tok:int) -> None:
        """
        Set alignment slider to a position corresponding to a given token index
        between 0 and self.num_encodings-1

        tok     the target token index (must be between 0 and self.num_tokens-1)
        """

        if tok >= 0 and tok < self.num_encodings:
            tok_as_sliderval = tok * self.attn_slider_resolution
            self.alignment_slider.setValue(tok_as_sliderval)
        else:
            print(f"Error: token index {tok} out of range (0-{self.num_encodings-1})")

    def get_slidervalue_as_params(self) -> np.ndarray:
        """slider value to attention parameters"""
        tok = self.alignment_slider.value() / self.attn_slider_resolution 

        # smoothing:
        tok = (tok + self.prev_tok) / 2
        width = abs(tok - self.prev_tok)
        self.prev_tok = tok
    
        return tok, width

    def addFrame(self, newframe:npt.ArrayLike, text:str):
        # TODO: discard frames which are out of display range
        # use a circular buffer instead of np.append
        if text != self.text:
            self.set_text(text)
        self.frame +=1
        start_step = max(0, self.frame-1-self.max_display_steps)
        self.imagedata = np.append(self.imagedata, newframe, axis=0)
        self.imageitem.setImage(image=self.imagedata)
        self.imageitem.update()
        self.plot.setYRange(start_step, self.frame-1, padding=0)

    def set_text(self, text):
        self.text = text

        ticks = [(i+0.5,c) for i,c in enumerate(text)]
        # print(ticks)
        self.plot.getAxis('top').setTicks([ticks, []])
        self.plot.getAxis('top').showLabel()
        # self.plot.getAxis('bottom').showLabel(True)
        self.plot.setXRange(0, len(text), padding=0)

    def reset(self, text:str=None):
        """
        Reset the attention graph with a given number of encoded tokens (y-axis)
        Usually called after new text is encoded / attention recalculated.

        num_encodings sets the number of encoded tokens to scale the y-axis by
            if not set the current number of encodings is left as-is
        """
        print(f"AlignmentGraph: reset")
        self.frame = 1
        if text is not None:
            self.num_encodings = len(text)
            self.set_text(text)
        self.attn_slider_max_value = (
            self.num_encodings * self.attn_slider_resolution)
        self.alignment_slider.setMinimum(0)
        self.alignment_slider.setMaximum(self.attn_slider_max_value)
        self.imagedata = np.zeros((1,self.num_encodings), dtype=np.float32)
        # self.imageitem.setImage(image=self.imagedata.T)
        self.plot.setYRange(0, 1, padding=0)
        self.imageitem.setImage(image=self.imagedata)
        self.imageitem.update()
        #self.plot.update()
        

class RaveLatents(QtWidgets.QWidget):
    def __init__(self, parent:QtWidgets.QWidget=None):
        super().__init__(parent)

        self.latents = list()
        self.layout = QtWidgets.QHBoxLayout(self)
        self.is_init = False

    def _init(self, num_latents, pitch_slider):
        for idx in range(num_latents):
            if idx==0 and pitch_slider:
                bmin, bmax = -100., 100.
                vmin, vmax = 50., 550.
            else:
                bmin, bmax = -3., 3.
                vmin, vmax = -5., 5.

            # latent widgets, each is a (SliderWidget, MeterWidget) pair
            # the first slide is bias, second is a display
            bias_slider = DoubleSlider(decimals=3, parent=self)
            bias_slider.setMaximum(bmax)
            bias_slider.setMinimum(bmin)
            bias_slider.setValue(0.0)
            bias_slider.doubleValueChanged.connect(
                lambda val,latent=idx: self._bias_adjust(val, latent))
            bias_slider.setStatusTip(f"bias latent dimension {idx}")
            
            value_meter = DoubleSlider(decimals=3, parent=self)
            value_meter.setMaximum(vmax)
            value_meter.setMinimum(vmin)
            value_meter.setValue(0.0)
            value_meter.setStyleSheet("""
                QSlider::groove:vertical {
                    background-color: black;
                    width: 10px;
                    border-radius: 5;
                }
                                      
                QSlider::add-page:vertical {
                    background: pink;
                    border-radius: 5;
                }
                QSlider::sub-page:vertical {
                    background: black;
                    border-radius: 5;
                }
                """)

            self.latents.append((bias_slider, value_meter))
            self.layout.addWidget(bias_slider)
            self.layout.addWidget(value_meter)
            if idx < num_latents-1:
                self.layout.addSpacing(12)
            self.layout.setSpacing(4)

        self.setMaximumHeight(200)

        self.is_init = True

    def _bias_adjust(self, val:float, latent:int):            
        # NOTE: The bias values of the sliders get sampled regularly in `update`, they are read by get_biases 
        print(f"ADJUST BIAS of LATENT {latent} = {val}")

    def get_bias(self, latent:int) -> float:
        return self.latents[latent][0].value()

    def get_biases(self) -> typing.List[float]:
    # def get_biases(self) -> npt.NDArray[np.float32]:
        # res = np.zeros((1,len(self.latents)), dtype=np.float32)
        # for idx, (bias, _) in enumerate(self.latents):
            # res[0][idx] = bias.value()
        # return res
        return [bias.value() for bias,_ in self.latents]
    
    def set_latent_bias(self, latent:int, value:float):
        """
        Set a latent bias value in the GUI
        """
        self.latents[latent][0].setValue(value)

    def add_latent_bias(self, latent:int, value:float):
        """
        Add a small value to the current bias value of a latent in the GUI
        """
        newval = self.latents[latent][0].value() + value;
        self.latents[latent][0].setValue(newval)

    def reset_latent_biases(self, value:float=0.0):
        """
        Reset all latent biases to 0.0
        """
        for bias,_ in self.latents:
            bias.setValue(value);

    def set_latents(self, values:typing.Union[list, npt.ArrayLike]):
        """
        Set latent values in the gui. 
        values > the number of sliders are ignored

        values  latent values as floats, in the shape (1, num_latents)
        """
        values = values[0] # trim off the extra dimension
        if len(values) <= len(self.latents):
            for idx,val in enumerate(values):
                self.latents[idx][1].setValue(val)
        else: # More values than sliders
            for idx,sliders in enumerate(self.latents):
                sliders[1].setValue(values[idx])

class ModelInfoDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, context:'MainWindow'=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Info")

        self.model_name = QtWidgets.QLabel()
        self.model_meta = QtWidgets.QLabel()

        self.setLayout(VBoxLayout(
            self.model_name,
            self.model_meta
        ))

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, context:'MainWindow'=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")

        def osc_listen_button_callback(val):
            result=None
            addr = (self._oschostline.text(), int(self._oscportline.text()))
            context.osc_listen_addr = addr
            status = self._osclistenbut.isChecked()
            if status:
                self._osclistenbut.setStyleSheet(
                    "color: black; background-color: lightblue")
                self._osclistenbut.setText("Listening..")
                # TODO: This use of MainWindow as context feels kludgy, should use an abstract class / mixin?
                if hasattr(context, "serve_osc"):
                    result = context.serve_osc(address=addr)
                else:
                    print("ERROR: No method 'serve_osc' defined on settings context.")
            else:
                self._osclistenbut.setStyleSheet(
                    "color: black; background-color: grey")
                self._osclistenbut.setText("Listen")
                if hasattr(context, "unserve_osc"):
                    result = context.unserve_osc()
                else:
                    print("ERROR: No method 'serve_osc' defined on settings context.")

            print(f"Listen button status: {status}")

        # Create GUI
        self._info = QtWidgets.QLabel("Settings...")

        self._osc = QtWidgets.QGroupBox(title="OSC Control Setup", parent=self)
        self._oschostline = QtWidgets.QLineEdit(context.osc_listen_addr[0])
        self._oschostline.setMaxLength(17)
        self._oscportline = QtWidgets.QLineEdit(str(context.osc_listen_addr[1]))
        self._oscportline.setMaxLength(5)
        self._osclistenbut = QtWidgets.QPushButton("&Listen")
        self._osclistenbut.setCheckable(True)
        self._osclistenbut.clicked.connect(osc_listen_button_callback)
        self._oscinfolabel = QtWidgets.QLabel("")
        hostportlayout = QtWidgets.QHBoxLayout()
        hostportlayout.addWidget(self._oschostline)
        hostportlayout.addWidget(QtWidgets.QLabel(":"))
        hostportlayout.addWidget(self._oscportline)
        hostport = QtWidgets.QWidget()
        hostport.setLayout(hostportlayout)
        osclayout = QtWidgets.QVBoxLayout()
        osclayout.addWidget(hostport)
        osclayout.addWidget(self._osclistenbut)
        osclayout.addWidget(self._oscinfolabel)
        self._osc.setLayout(osclayout)
        
        OKCLOSE = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self._buttonBox = QtWidgets.QDialogButtonBox(OKCLOSE)
        self._buttonBox.accepted.connect(self.accept)
        self._buttonBox.rejected.connect(self.reject)

        self._oscinstructions = QtWidgets.QLabel(context.osc_controller.print_osc_api())
        self._oscinstructions.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._oscinstructions_scroller = QtWidgets.QScrollArea()
        self._oscinstructions_scroller.setWidget(self._oscinstructions)

        self._main_layout = VBoxLayout(
            self._info,
            self._osc,
            self._buttonBox,
            QtWidgets.QFrame(),
            self._oscinstructions_scroller
        )

        self.setLayout(self._main_layout)

# Glue between independent threads and the main Qt GUI
class Signaller(QtCore.QObject) :
    # List signals here
    set_gen_text = QtCore.Signal(str)
    set_samp_text = QtCore.Signal(str)

signaller = Signaller()

# Controller thread that receives OSC messages and maps them to Tungnaa actions
class OSCController(threading.Thread):

    def __init__(self, context:'MainWindow', *args, **kwargs):
        super(OSCController, self).__init__(*args, **kwargs)
        self.context = context
        self.osc_server = None
        self.osc_dispatcher = None    

        # BEGIN: OSC Dispatcher Callback Functions -----------------------------
        def osc_unknown(addr:str, *args:list[typing.Any]) -> None:
            print(f"Unknown OSC address: {addr}  with: '{args}'")

        def osc_generate(addr:str, reset:bool=True) -> None:
            """Run/Play generator mode
            reset   bool if true reset to beginning of utterance before playback (useful if generate_stop_at_end is active)
            """
            # TODO: maybe adds some timing wierdness having to go through the QtGui thread for the reset..?
            if reset:
                self.context._reset_action.trigger()
            self.context._generate_action.trigger()

        def osc_sampler(addr:str, *args:list[typing.Any]) -> None:
            """(trigger)Activate sampler mode"""
            self.context._sampler_action.trigger()

        def osc_pause(addr:str, *args:list[typing.Any]) -> None:
            """(trigger)Pause generator/sampler playback"""
            self.context._pause_action.trigger()

        def osc_reset(addr:str, *args:list[typing.Any]) -> None:
            """(trigger)Reset generator/sampler playback"""
            self.context._reset_action.trigger()

        def osc_set_bias(addr:str, latent:int, bias:float) -> None:
            """Set vocoder latent bias
            latent      int index of latent
            bias        float bias amount (usually not larger than +-3.0)

            """
            self.context.latents.set_latent_bias(latent, bias)
            print(f"Set latent {latent} bias: {bias}")

        def osc_add_bias(addr:str, latent:int, bias:float) -> None:
            """Set add a value to the current vocoder latent bias
            latent      int index of latent
            bias        float amount to add to bias (usually a small value, will clip if larger than +-3.0)

            """
            self.context.latents.add_latent_bias(latent, bias)
            print(f"Add {bias} to latent {latent} bias")


        def osc_reset_biases(addr:str) -> None:
            """Reset all biases to 0.0"""
            self.context.latents.reset_latent_biases()
            print(f"Reset latent biases to 0.0")


        def osc_generate_stop_at_end(addr:str, enable:bool) -> None:
            """ Enable/disable Generator stop-at-end of utterance
            enable     bool true or false
            """
            self.context._generate_stop_at_end_toggle_action.setChecked(enable)

        def osc_alignment_mode(addr:str, mode:str) -> None:
            """Set alignment generation mode (Generator only)
            mode    str 'infer' for alignment inference, 'paint' for alignment painting
            """
            self.context.set_alignment_mode(mode)

        def osc_latent_feedback(addr:str, enable:bool) -> None:
            """Enable/disable latent feedback mode(Generator only)
            enable     bool true or false
            """
            self.context._latent_feedback_toggle_action.setChecked(enable)


        def osc_set_gen_text(addr:str, text:str, encode:bool=True) -> None:
            """Set generator utterance text in gui textbox (Generator only)
            text        str new generate utterance text
            encode      bool if true, automatically encode the new text utterance
            """
            print(f"OSCfunc: set_gen_text {addr} | {text=} | {encode=}")
            # NOTE: One of the main gotcha's about Qt is that you cannot call any QWidget methods from any 
            # thread other than the main GUI thread. All of your communication must be done by emitting signals 
            # from the extra threads, which will forward to the main gui. So the following line will not work!
            # self.text_input.setText(text)
            signaller.set_gen_text.emit(text)

            if encode:
                print("ENCODE TEXT")
                self.context.btn_send_gen_text.click()

        def osc_set_alignment_as_token_idx(addr:str, tok_idx:int, force_paint:bool=False) -> None:
            """Set alignment of generator by token index (Generator Only)
            token_idx           int index of token
            force_paint         bool if true, toggles on attention painting
            """
            self.context.attention_graph.set_alignment_as_token_idx(tok_idx)
            print(f"Set alignment to token {tok_idx} - forced paint?: {force_paint}")

        def osc_set_alignment_normalized(addr:str, normalized_align:float, force_paint:bool=False) -> None:
            """Set alignment of generator by a normalized 0.0-1.0 value (Generator Only)
            normalized_align    float alignment value from 0-1 gets mapped to start-end token range
            force_paint         bool if true, toggles on attention painting
            """
            self.context.attention_graph.set_normalized_alignment(normalized_align)
            print(f"Set normalized alignment {normalized_align} - forced paint?: {force_paint}")

        def osc_set_temperature(addr:str, temp:float) -> None:
            """Set generator sampling temperature (Generator Only)
            temp    float temperature from 0.0-1.0 (can go up to 2.0 for more weird predictions)

            """
            self.context.set_temperature(temp)
            print(f"Set sampling temp {temp}")

        def osc_add_temperature(addr:str, temp:float) -> None:
            """Add a small value to the generator sampling temperature (Generator Only)
            temp    float value to add to temperature (usually below 1.0) - will clip at 0 and a max value

            """
            self.context.add_temperature(temp)
            print(f"Add {temp} to sampling temp")

        def osc_sampler_stop_at_end(addr:str, enable:bool) -> None:
            """ Enable/disable Sampler stop-at-end loop point
            enable     bool true or false
            """
            self.context._sampler_stop_at_end_toggle_action.setChecked(enable)

        def osc_set_sampler_step(addr:str, step:int, autoplay:bool=True):
            """Set sampler playhead absolute position (Sampler only)
            step        int absolute index of vocoder frame buffer to set sampler playhead, index wraps and can be negative
            autoplay    bool if true, sampler playback is triggered if it is not already playing
            """
            print(f"OSCfunc: set_sampler_step {addr} | {step=}")
            self.context.sampler_step(step=step, autoplay=autoplay)

        def osc_set_loop_text(addr:str, 
                text:str, n:int=-1, 
                start:bool=True, end:bool=True, reset:bool=True
                ) -> None:
            """Set sampler loop points to nth occurrence of matched text (Sampler only)
            text            str text to match in sampler history, can be a single token
            n               int which of the n text matches to loop, -1 is most recent, 0 oldest, etc..
            start           bool if true, updates the sampler loop start at the matched text
            end             bool if true, updates the sampler loop end at the matched text
            reset           bool if true resets sampler playhead to start of text match        
            """
            print(f"OSCfunc: set_loop_text {addr} | {text=} | {n=} | {start=} | {end=} | {reset=}")
            signaller.set_samp_text.emit(text)

            print("SET LOOP")
            self.context.loop_text(text=text, n=n, start=start, end=end, reset=reset)

        def osc_set_loop_index(addr:str, 
                start:int=None, end:int=None, 
                reset:bool=True,
                utterance:int=None,
                ) -> None:
            """Set sampler loop points by vocoder frame with option to index by utterance (Sampler only)
            start           int global start frame index
            end             int global end frame index
            reset           bool if true resets sampler playhead to start index
            utterance       global utterance index, if supplied changes start and end to be utterance-relative


            """
            print(f"OSCfunc: set_loop_index {addr} | {start=} | {end=} | {reset=} | {utterance=}")
            print("SET LOOP")
            self.context.loop_index(start=start, end=end, utterance=utterance, reset=reset)


        # END:: OSC Dispatcher Callback Functions -----------------------------------------------------------------------


        self.osc_dispatcher = pythonosc.dispatcher.Dispatcher()
        self.osc_dispatcher.set_default_handler(osc_unknown)
        
        # OSC Callback mappings
        self.osc_dispatcher.map("/generate", osc_generate) # TODO rename
        self.osc_dispatcher.map("/sampler", osc_sampler)
        self.osc_dispatcher.map("/pause", osc_pause)
        self.osc_dispatcher.map("/reset", osc_reset)

        self.osc_dispatcher.map("/set_bias", osc_set_bias)
        self.osc_dispatcher.map("/add_bias", osc_add_bias)
        self.osc_dispatcher.map("/reset_biases", osc_reset_biases)

        self.osc_dispatcher.map("/generate_stop_at_end", osc_generate_stop_at_end)
        self.osc_dispatcher.map("/alignment_mode", osc_alignment_mode)
        self.osc_dispatcher.map("/latent_feedback", osc_latent_feedback)
        self.osc_dispatcher.map("/set_gen_text", osc_set_gen_text)
        self.osc_dispatcher.map("/set_token", osc_set_alignment_as_token_idx)
        self.osc_dispatcher.map("/set_alignment", osc_set_alignment_normalized)
        self.osc_dispatcher.map("/set_temperature", osc_set_temperature)
        self.osc_dispatcher.map("/add_temperature", osc_add_temperature)

        self.osc_dispatcher.map("/sampler_stop_at_end", osc_sampler_stop_at_end)
        self.osc_dispatcher.map("/set_sampler_step", osc_set_sampler_step)
        self.osc_dispatcher.map("/set_loop_text", osc_set_loop_text)
        self.osc_dispatcher.map("/set_loop_index", osc_set_loop_index)



    def run(self):
        # Thread entry point
        print(f"Serving on {self.osc_server.server_address}")
        self.osc_server.serve_forever()
        print(f"Closing OSC Server...")

        # Need to update the GUI
        # signaller.set_text.emit("OSC Server Listening")


    def unserve_osc(self):
        if self.osc_server is not None:
            self.osc_server.shutdown()
            self.osc_server.server_close()
            self.osc_server = None
            print("Waiting for server to shutdown...")
            #self.osc_server_thread.join()
            print("Server Thread closed...")

    def serve_osc(self, address=("localhost", 7777)):

        if self.osc_server is not None:
            self.unserve_osc()

        self.osc_server = pythonosc.osc_server.ThreadingOSCUDPServer(address, self.osc_dispatcher)
        self.daemon = True
        #self.osc_server_thread = threading.Thread(target=run_osc_server, args=(self,), daemon=True)
        self.start()
        print(f"OSC Server Thread Started")


    # TODO can be improved to be co-generated together with dispatcher mappings
    def print_osc_api(self, basepath='/tungnaa'):
        instructions=""

        # TODO can be improved, this approach depends on undocumented internals of pythonosc.dispatcher
        if self.osc_dispatcher is not None:
            instructions += "OSC Control API:\n"
            for addr in self.osc_dispatcher._map.keys():
                handler = self.osc_dispatcher._map[addr][0] # take first available handler, although multiple are possible
                print(f"{addr} - {handler}")
                instructions += f"{addr} - {handler.callback.__doc__} \n" 

        instructions += f"""OSC Out API:
        {basepath}/latents
        {basepath}/status"""
        return instructions



class MainWindow(QtWidgets.QMainWindow):
    """
    Main Application Window
    """

    def __init__(self, 
        parent:QtWidgets.QWidget=None, 
        use_backend:bool=True, 
        backend:tungnaa.gui.backend.Backend=None, 
        sender:tungnaa.gui.senders.Sender=None,
        update_fps:int=100,
        osc_listen_addr:tuple[str,int]=('localhost', 1337),
        stress_gui=None,
        text=None,
        sampler_text=None,
        ):
        super().__init__(parent)
        self.version= f"Alpha v{tungnaa.__version__}"
        self.appname= f"T̴u̮n̵g̴na͠á {self.version}"

        self.stress_gui = stress_gui

        if sys.platform == 'darwin':
            self.setUnifiedTitleAndToolBarOnMac(True)

        self.mode = 'infer'
        
        self.update_fps = update_fps
        self.osc_listen_addr = osc_listen_addr
        self.use_backend = use_backend
        self.backend = backend
        self.sender = sender
        if self.backend is None:
            print("No backend provided, disabling backend")
            self.use_backend = False
        
        # OSC
        self.osc_controller = OSCController(context=self)
        
        # TODO: this line auto-runs the OSC server and creates a bug
        #     Currently needs to be activated from the gui, running this line starts the OSC server but does not reflect this status in the gui
        #self.serve_osc(tuple(osc_listen_addr))

        # Signals used by other threads to communicate with GUI objects defined in the main thread.

        # ---------------------------------------------------------------------
        # BUILD GUI
        # ---------------------------------------------------------------------
        #self.setWindowTitle(QtCore.QCoreApplication.applicationName())
        
        self.setWindowTitle(self.appname)
        self.main = QtWidgets.QWidget(self)
        # self.main.setStyleSheet("color: grey; background-color: black")
        self.setCentralWidget(self.main)
        self.settings_dialog = None # Settings modal dialogue
        self.app_icon = QtGui.QIcon()
        with resources.path("tungnaa.resources", "tungnaa_icon.png") as path:
            self.app_icon.addFile(path.as_posix())
        self.setWindowIcon(self.app_icon)
        self.setWindowIconText(self.appname)

        self.tungnaa_logo = QtWidgets.QLabel(parent=self.main)
        with resources.path("tungnaa.resources", "tungnaa_logo.png") as path:
            logo_pixmap = QtGui.QPixmap(path.as_posix())
        self.tungnaa_logo.setPixmap(logo_pixmap)
        self.tungnaa_logo.setMargin(1)
        self.tungnaa_logo.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        # self.tungnaa_logo.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(
        #     xOffset=0, yOffset=0, blurRadius=20, 
        #     color=QtGui.QColor(0, 0, 0)))

        self.tungnaa_version = QtWidgets.QLabel(parent=self.main)
        self.tungnaa_version.setText(self.version)
        version_font = QtGui.QFont()
        version_font.setPointSize(8)
        self.tungnaa_version.setFont(version_font)
        self.tungnaa_version.setMargin(0)
        self.tungnaa_version.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)

        feedback_url = "https://forms.gle/F97yhJ1YB5aiZPmn7"
        self.feedback_link = QtWidgets.QLabel(parent=self.main)
        self.feedback_link.setText(f'<a href="{feedback_url}">feedback...</a>')
        print(f'we would love your feedback on Tungnaá in this short survey: {feedback_url}')
        self.feedback_link.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.feedback_link.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextBrowserInteraction)
        self.feedback_link.setOpenExternalLinks(True)
        version_font = QtGui.QFont()
        version_font.setPointSize(10)
        self.feedback_link.setFont(version_font)
        self.feedback_link.setMargin(0)
        self.feedback_link.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop)
        self.feedback_link.setStatusTip("link to a short survey form")

        # Main App Toolbar
        self.toolbar_top = QtWidgets.QToolBar(self.main)
        self.toolbar_combined = QtWidgets.QWidget(self.main)
        self.toolbar_controls = QtWidgets.QToolBar(self.toolbar_combined)
        controls_layout = HBoxLayout()
        controls_layout.addWidget(self.toolbar_controls)
        controls_layout.addStretch(1)
        # controls_layout.addLayout(VBoxLayout(self.tungnaa_logo, self.tungnaa_version, self.feedback_link))
        controls_layout.addLayout(HBoxLayout(
            self.tungnaa_logo, 
            VBoxLayout(self.feedback_link, self.tungnaa_version)))
        self.toolbar_combined.setLayout(controls_layout)
        self.addToolBar(self.toolbar_top) # Toolbar gets added to VLayout below

        self._generate_action = QtGui.QAction("Generate", self)
        self._generate_action.setStatusTip("Start autoregression")
        self._generate_action.triggered.connect(self.play_generate)

        self._sampler_action = QtGui.QAction("Sampler", self)
        self._sampler_action.setStatusTip("Sample generated latents")
        self._sampler_action.triggered.connect(self.play_sampler)

        self._pause_action = QtGui.QAction("Pause", self)
        self._pause_action.setStatusTip("Pause generation or sampler")
        self._pause_action.triggered.connect(self.pause)

        self._reset_action = QtGui.QAction("Reset", self)
        self._reset_action.setStatusTip("Reset autoregression history")
        self._reset_action.triggered.connect(self.reset_autoregression)

        self._alignment_paint_toggle_action = QtGui.QAction("Attention Painting", self)
        self._alignment_paint_toggle_action.setCheckable(True)
        self._alignment_paint_toggle_action.setStatusTip("Toggle Attention Painting")
        self._alignment_paint_toggle_action.toggled.connect(self.toggle_alignment_paint)

        self._latent_feedback_toggle_action = QtGui.QAction("Latent Feedback", self)
        self._latent_feedback_toggle_action.setCheckable(True)
        self._latent_feedback_toggle_action.setStatusTip("Feed latent manipulation back to model")
        self._latent_feedback_toggle_action.toggled.connect(self.toggle_latent_feedback)

        self._generate_stop_at_end_toggle_action = QtGui.QAction("Generator Stop At End", self)
        self._generate_stop_at_end_toggle_action.setCheckable(True)
        self._generate_stop_at_end_toggle_action.setStatusTip("Pause automatically at end of utterance text in generate mode")
        self._generate_stop_at_end_toggle_action.toggled.connect(self.toggle_generate_stop_at_end)

        self._sampler_stop_at_end_toggle_action = QtGui.QAction("Sampler Stop At End", self)
        self._sampler_stop_at_end_toggle_action.setCheckable(True)
        self._sampler_stop_at_end_toggle_action.setStatusTip("Pause automatically at loop end point in sampler mode")
        self._sampler_stop_at_end_toggle_action.toggled.connect(self.toggle_sampler_stop_at_end)

        self._latent_bias_reset_action = QtGui.QAction("Bias Reset", self)
        self._latent_bias_reset_action.setStatusTip("Reset all latent biases to 0.0")
        self._latent_bias_reset_action.triggered.connect(self._reset_latent_biases)

        self._settings_action = QtGui.QAction("Settings", self)
        self._settings_action.setStatusTip("Adjust OSC, MIDI and Audio settings")
        self._settings_action.triggered.connect(self.open_settings)

        self._model_info_action = QtGui.QAction("Model Info", self)
        self._model_info_action.setStatusTip("Show model metadata")
        self._model_info_action.triggered.connect(self.open_model_info)
        self.model_info_dialog = ModelInfoDialog(context=self)

        self.temperature_slider = DoubleSlider(orientation=QtCore.Qt.Horizontal, decimals=3, parent=self)
        self.temperature_slider.setMaximum(2.0)
        self.temperature_slider.setMinimum(0.0)
        self.temperature_slider.setValue(1)
        self.temperature_slider.doubleValueChanged.connect(lambda val: self._temperature_adjust(val))
        self.temperature_slider.setStatusTip("sampling temperature")


        self.toolbar_top.addAction(self._alignment_paint_toggle_action)
        self.toolbar_top.addAction(self._latent_feedback_toggle_action)
        self.toolbar_top.addAction(self._sampler_stop_at_end_toggle_action)
        self.toolbar_top.addAction(self._generate_stop_at_end_toggle_action)
        self.toolbar_top.addAction(self._settings_action)
        self.toolbar_top.addAction(self._model_info_action)

        self.toolbar_controls.addAction(self._generate_action)
        self.toolbar_controls.addAction(self._sampler_action)
        self.toolbar_controls.addAction(self._pause_action)
        self.toolbar_controls.addAction(self._reset_action)
        self.toolbar_controls.addAction(self._latent_bias_reset_action)
        
        # Tabbed Interface for Generator/Sampler
        self.tabwidget = QtWidgets.QTabWidget(self.main);

        # Generator Tab
        self.generator_widget = QtWidgets.QWidget(self.tabwidget);

        # See https://doc.qt.io/qt-6/qtextedit.html for all the fun slots for textedit
        self.gen_text_input = QtWidgets.QPlainTextEdit(parent=self.generator_widget)
        self.gen_text_input.setMaximumHeight(100)
        self.gen_text_input.setFont(QtGui.QFont("Arial", 24))
        self.gen_text_input.setPlainText(text or "<It took me a while to have a voice. And now that I have one, I am not going to be silent.>")
        self.gen_text_input.setStatusTip("enter text to generate new utterances")
        signaller.set_gen_text.connect(self.gen_text_input.setPlainText)
        #self.text_input.setPlainText("It took me a while to find a voice, and now that I have one, I am not going to be silent.") # todo - add a "random phrase" generator?
        self.btn_send_gen_text = QtWidgets.QPushButton("Encode", parent=self.generator_widget)
        self.btn_send_gen_text.clicked.connect(self.encode_text)
        self.btn_send_gen_text.setStatusTip("send text to model")
        self.text_encoding_status = QtWidgets.QLabel(parent=self.generator_widget)
        self.text_encoding_status.setText("ʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ encoder feedback... ʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ")


        self.attention_graph = AlignmentGraph(parent=self.generator_widget, image_clicked=self.image_clicked)
        self.attention_graph.setStatusTip("text-audio alignments (click to jump)")
        
        self._generator_layout = VBoxLayout(
                self.gen_text_input, 
                self.btn_send_gen_text,
                self.text_encoding_status,
                self.attention_graph,
                spacing=0, margins=(0,0,0,0)
        )
        self.generator_widget.setLayout(self._generator_layout);


        # Sampler Tab
        self.sampler_widget = QtWidgets.QWidget(self.tabwidget);

        self.samp_text_input = QtWidgets.QPlainTextEdit(self.sampler_widget)
        self.samp_text_input.setMaximumHeight(100)
        self.samp_text_input.setFont(QtGui.QFont("Arial", 24))
        self.samp_text_input.setPlainText(sampler_text or "took")
        self.samp_text_input.setStatusTip("enter text to search through previous utterances in sampler mode")
        signaller.set_samp_text.connect(self.samp_text_input.setPlainText)
        self.btn_send_samp_text = QtWidgets.QPushButton("Sample", parent=self.sampler_widget)
        self.btn_send_samp_text.clicked.connect(self.loop_text)
        self.btn_send_samp_text.setStatusTip("set sampler loop points by text")


        self._sampler_layout = VBoxLayout(
            VBoxLayout(
                self.samp_text_input, 
                self.btn_send_samp_text,
                spacing=0, margins=(0,0,0,0)
            ),
            spacing=0, margins=(12,0,12,0),
        )
        self.sampler_widget.setLayout(self._sampler_layout)

        self.tabwidget.addTab(self.generator_widget, "Generator")
        self.tabwidget.addTab(self.sampler_widget, "Sampler")        



        # StatusBar
        self.statusbar = QtWidgets.QStatusBar(self.main)
        self.setStatusBar(self.statusbar)

        self.latents = RaveLatents(self.main)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonFollowStyle)


        self._main_layout = VBoxLayout(
            self.toolbar_combined,
            self.tabwidget,
            self.latents,
            self.temperature_slider,
            spacing=0, margins=(0,0,0,0)
        )
        self.main.setLayout(self._main_layout)

        # Keyboard Shortcuts (TODO: put these in a menu somewhere)
        self._encode_action = QtGui.QAction("Encode", self)
        #self._encode_action.autoRepeat = False
        self._encode_action.triggered.connect(self.btn_send_gen_text.click)
        self._encode_action.setShortcut("Ctrl+Return") # TODO: Shift+Return overridden by QTextEdit ... need to subclass?
        self.addAction(self._encode_action) # add to the main window as a global shortcut

        self.gui_update_timer = QtCore.QTimer(self)
        self.gui_update_timer.timeout.connect(self.update)
        
        # Used to generate fake data in update() when no backend is provided
        self.frame = 0
        self.tok = 30
        self.max_tok = 40
        self.gui_update_timer.start((1.0 / self.update_fps) * 1000)

        self._generate_stop_at_end_toggle_action.setChecked(False) # make the default to babble like a river
        self._sampler_stop_at_end_toggle_action.setChecked(True)

    def set_metadata(self, name, meta):
        if name is not None:
            self.model_info_dialog.model_name.setText(name)
        if meta is not None:
            meta_str = ""
            for k,v in meta.Meta.items():
                # if k!='vocoder':
                meta_str += f'{k}: {v}\n'
            self.model_info_dialog.model_meta.setText(meta_str)

    def image_clicked(self, ev):
        # print(ev.pos)
        self.backend.set_momentary_alignment((ev.pos().x(), 1))
        self.backend.set_state_by_step(int(ev.pos().y()))
        self.backend.generate()
        # self.backend.set_momentary_alignment(
        #     self.attention_graph.get_slidervalue_as_params())

    def update(self):
        """
        update method runs on a timer
        empties the queue from the backend and updates gui elements
        """
        self.frame += 1
        new_data = list()
        # empty message queue, update attention graph & RAVE latents
        if self.use_backend: 

            if self.stress_gui is not None:
                t = time.time_ns()
                while time.time_ns() - t < self.stress_gui*1e9:
                    pass

            try:
                if self.latents.is_init:
                    self.backend.set_biases(self.latents.get_biases())
                if self.mode == 'paint':
                    self.backend.set_alignment(
                        self.attention_graph.get_slidervalue_as_params())
                elif self.mode == 'infer':
                    self.backend.set_alignment(None)
                else:
                    raise ValueError(f"Unknown attention mode: {self.mode} - must be <infer|paint>")

                # for _ in range(self.backend.frontend_q.qsize()):
                while self.backend.parent_conn.poll():
                    try:
                        # framedict = self.backend.frontend_q.get_nowait()
                        framedict = self.backend.parent_conn.recv()
                        if not self.latents.is_init and 'num_latents' in framedict and 'use_pitch' in framedict:
                            self.latents._init(
                                framedict['num_latents'], framedict['use_pitch'])

                        new_data.append(framedict)
                    except queue.Empty as ex:
                        break
            except Exception as e:
                print(e)
                exit(0) ### debug
            
        else: # Do not use backend, instead generate random data.. useful for testing the gui (maybe?)
            new_attn_frame = np.zeros((1,self.attention_graph.num_encodings), dtype=np.float32)        
            new_latent_frame = np.random.rand(1,8) * 0.5
            new_latent_frame = new_latent_frame + self.latents.get_biases()
            if self.mode == 'infer':
                if random.random() < 0.2:
                    self.tok += random.choice([-1, 1])
            elif self.mode == 'paint':
                self.tok = self.attention_graph.alignment_slider.value()            
            else:
                raise ValueError(f"Unknown attention mode: {self.mode} - must be <infer|paint>")

            if self.tok >= self.max_tok-1:
                self.tok = self.max_tok - 1
            else:
                new_attn_frame[0,self.tok + 1] = 0.5
            if self.tok <= 0:
                self.tok = 0
            else:
                new_attn_frame[0,self.tok - 1] = 0.5
            new_attn_frame[0,self.tok] = 1.0

            new_data.append({
                'latent_t': new_latent_frame, 'align_t': new_attn_frame})
            
        # iterate through new_data and update the gui
        for datadict in new_data:
            if datadict.get('reset', False):
                # self.finish_reset(num_tokens=datadict['align_t'].shape[-1])
                self.finish_reset(text=datadict['text'])
            if 'align_t' in datadict and not datadict.get('sampler', False):
                self.attention_graph.addFrame(
                    datadict['align_t'], datadict['text'])
        
        latents = [d['latent_t'] for d in new_data if 'latent_t' in d]
        if len(latents) and self.latents.is_init:
            self.latents.set_latents(values=latents[-1])

    def finish_reset(self, text):
        """
        Called when backend signals a reset.
        """
        self.text_encoding_status.setText(f"ʭʬʭʬʭʬʭʬ encoded {len(text)} embeddings ʬʭʬʭʬʭʬʭ")
        self.attention_graph.reset(text=text)

    def closeEvent(self, e:QtCore.QEvent):
        """
        Cleanup
        """
        # TODO: cleanup OSC/networking connections
        print(f"Application Close {e}")
        self.backend.cleanup()
        e.accept()
        #e.ignore() # Under some conditions ignore app close?

    def _temperature_adjust(self, temp=1.0) -> None:
        """
        Private method adjust model step inference temperature
        """
        if self.use_backend:
            self.set_temperature(temp)

    def _reset_latent_biases(self):
        """Private method to reset latent biases to 0.0"""
        if self.latents.is_init:
            self.latents.reset_latent_biases(0.0)

    def encode_text(self):
        """
        Send input text to the text encoder backend
        """
        if self.use_backend:
            txtval = self.gen_text_input.toPlainText()
            print(f"Encoding: {txtval}")
            self.text_encoding_status.setText("ʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ encoding.... ʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭʬʭ")
            # this returns the text with start/end tokens added and loop points stripped
            self.backend.set_text(text=txtval)
        else:
            print("No backend enabled to encode text: ignoring...")

    def sampler_step(self, step:int, autoplay:bool) -> None:
        self.backend.set_sampler_step(step)
        if autoplay:
            # TODO: Once sampler mode is a toggle rather than a trigger, need to implement something like this to put the GUI in sampler mode...
            # if gui not_in_sampler_mode                
            #     self._sampler_action.toggle(True)
            # else:
            self.backend.sampler()

    def loop_text(self, **kw):
        """
        Send input text to the sampler backend
        """
        if not self.use_backend:
            print("No backend enabled to loop text: ignoring...")
            return
        if 'text' not in kw:
            kw['text'] = self.samp_text_input.toPlainText()
        self.backend.set_sampler_loop_text(**kw)

    def loop_index(self, **kw):
        """
        Send input text to the sampler backend
        """
        if not self.use_backend:
            print("No backend enabled to loop indices: ignoring...")
            return
        self.backend.set_sampler_loop_index(**kw)

    def play_generate(self, val:bool):
        print(f"Play autoregressive frame generator")
        ### TODO: replace this
        # if self.backend.text_rep is None:
            # self.encode_text()
        if self.attention_graph.text is None:
            self.encode_text()
        self.backend.generate()

    def play_sampler(self, val:bool):
        print(f"Play sampler")
        self.backend.sampler()

    def pause(self, val:bool):
        print(f"Pause generation or sampler")
        self.backend.pause()

    def reset_autoregression(self, val:bool):
        print(f"Reset autoregression history")
        # self.attention_graph.reset()
        self.backend.reset()

    def set_temperature(self, temp:float) -> None:
        """
        Set inference temperature (used by OSC/MIDI)

        Args:
            temp  inference temperature from 0.0-2.0
        """
        if temp > 2.0:
            temp=2.0
        elif temp < 0:
            temp=0
        self.temperature_slider.setValue(temp)
        self.backend.set_temperature(temp)

    def add_temperature(self, temp:float) -> None:
        """
        Add a small value to the inference temperature (used by OSC/MIDI)

        Args:
            temp    value that will be added to inference temperature, usually small, temperature will clip at [0,2]
        """
        self.set_temperature(self.temperature_slider.value() + temp)

    def toggle_alignment_paint(self, toggle:bool):
        if toggle:
            self.mode = 'paint'
        else:
            self.mode = 'infer'
        print(f"Alignment Mode Changed To:{self.mode}")

    def toggle_latent_feedback(self, toggle:bool):
        self.backend.set_latent_feedback(toggle)
        print(f"Latent Feedback status changed to:{toggle}")

    def toggle_generate_stop_at_end(self, toggle:bool):
        self.backend.set_generate_stop_at_end(toggle)
        print(f"Stop Generate At End status changed to:{toggle}")

    def toggle_sampler_stop_at_end(self, toggle:bool):
        self.backend.set_sampler_stop_at_end(toggle)
        print(f"Stop Sampler At End status changed to:{toggle}")

    def set_alignment_mode(self, mode:str='infer') -> None:
        """
        Set alignment mode directly (used by OSC/MIDI)
        """
        if mode in ['infer', 'paint']:
            self.mode = mode
            self._alignment_paint_toggle_action.setChecked((mode == 'paint'))
            print(f"Alignment Mode changed to: {mode}")

    def open_settings(self):
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(context=self)
        self.settings_dialog.show() # use show() to display a modeless dialog

    def open_model_info(self):
        self.model_info_dialog.show() # use show() to display a modeless dialog

    def serve_osc(self, address=("localhost", 7777)):
        self.osc_controller.unserve_osc()
        self.osc_controller.serve_osc(address=address)

    def unserve_osc(self):
        self.osc_controller.unserve_osc()

# Main entry point for Fire
def main(
    # models
    tts:Path='tungnaa_119_vctk',
    vocoder:Path=None,
    repo='Intelligent-Instruments-Lab/tungnaa-models-public',
    # output modes
    synth_audio:bool=True,
    latent_audio:bool=False,
    latent_osc:bool=False,
    # audio driver
    audio_out:typing.Union[str,int]=None,
    audio_block:int=2048,
    # OSC
    osc_out_addr:str='localhost:57120',
    osc_out_path:str='/tungnaa',
    osc_out_scsynth:bool=False,
    osc_in_addr:str='localhost:1337',
    # misc
    no_backend:bool=False,
    buffer_frames:int=1,
    stress_gui:float=None,
    jit:bool=False,
    profile:bool=False,
    text:str|None=None,
    sampler_text:str|None=None,
):
    """Tungnaa Text to Voice
    
    Args:
        tts: path to TTS model, or name of model in repo
        vocoder: path to vocoder model
        repo: huggingface repo to search for models

        synth_audio: send stereo audio from Python
        latent_audio: pack latents into a mono audio signal, 
            to be unpacked into multiple signals and decoded elsewhere
        latent_osc: send latents over OSC, 
            to be converted to audio signals and decoded elsewhere

        audio_out: 'default', or audio output device number (see 'tungnaa device' for a list of devices)
        audio_block: audio block size in samples

        osc_out_addr: host:port to send OSC data to. 
            By default sends to localhost:57120
        osc_out_path: The OSC path prefix of messages sent by this app. 
            Uses /tungnaa by default.
        osc_out_scsynth: format OSC for sending directly to scsynth
        osc_in_addr: host:port for recieving OSC control messages from clients.   
            By default listens on localhost:1337
        buffer_frames: compute ahead this many model steps
            increases latency, but may mitigate dropouts
        stress_gui: for testing, add this many seconds delay in `update`
        
    """

    if audio_out == 'default':
        audio_out = None
    if audio_out is not None:
        if isinstance(audio_out, str) and audio_out.isdecimal():
            audio_out = int(audio_out)
        elif not isinstance(audio_out, int):
            raise TypeError(f"Invalid audio device id '{audio_out}', must be either an integer or 'default'")

    app = QtWidgets.QApplication(sys.argv)
    QtCore.QCoreApplication.setOrganizationName("Intelligent Instruments Lab")
    QtCore.QCoreApplication.setApplicationName("Tungnaa")
    QtCore.QCoreApplication.setApplicationVersion(QtCore.qVersion())

    if latent_osc:
        if osc_out_scsynth: # Send latents to scsynth via OSC using default scsynth address localhost:57110
            sender = tungnaa.gui.senders.SCSynthDirectOSCSender(
                host='127.0.0.1',
                port=57110,
                bus_index=64,
                latency=0.2
            )
        else: # Send latents out via Generic OSC
            osc_host, osc_port = osc_out_addr.split(':')
            osc_port = int(osc_port)
            sender = tungnaa.gui.senders.GenericOSCSender(
                host=osc_host, 
                port=osc_port, 
                lroute=f'{osc_out_path}/latents',
                sroute=f'{osc_out_path}/status'
            )
    else:
        sender = None

    # if tts is not a local file, download model from repo
    # also sets the vocoder unless it it explicitly set to something else
    model_name, model_meta = None, None
    try:
        with open(tts): pass
    except FileNotFoundError:
        print(f'searching remote repo for model "{tts}"...')
        tts, vocoder, model_name, model_meta = dl_model(repo, tts, vocoder)

    backend = tungnaa.gui.backend.Backend(
        checkpoint=tts, 
        rave_path=vocoder,
        audio_out=audio_out,
        audio_block=audio_block,
        # audio_channels=audio_channels,
        synth_audio=synth_audio,
        latent_audio=latent_audio,
        latent_osc=latent_osc,
        osc_sender=sender,
        buffer_frames=buffer_frames,
        jit=jit,
        profile=profile,
        )
    
    osc_host, osc_port = osc_in_addr.split(':')
    osc_port = int(osc_port)
    win = MainWindow(
        use_backend=(not no_backend), 
        sender=sender,
        backend=Proxy(backend),
        osc_listen_addr=(osc_host, osc_port),
        stress_gui=stress_gui,
        text=text,
        sampler_text=sampler_text,
        )

    available_geometry = win.screen().availableGeometry()
    win.resize(available_geometry.width() / 2 * 1.063, available_geometry.height())
    win.move((available_geometry.width() - win.width()), 0)
    win.show()

    win.set_metadata(model_name, model_meta)

    # backend.start_stream()

    sys.exit(app.exec())

if __name__ == "__main__":
    fire.Fire(main)
