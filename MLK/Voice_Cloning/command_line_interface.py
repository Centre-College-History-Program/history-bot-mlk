from .toolbox.utterance import Utterance
from pathlib import Path
from typing import List, Set
import sounddevice as sd
import numpy as np
# from sklearn.manifold import TSNE         # You can try with TSNE if you like, I prefer UMAP 
from time import sleep
import umap
import sys
from warnings import filterwarnings
from .encoder import inference as encoder
from .synthesizer.inference import Synthesizer
from .vocoder import inference as vocoder
from pathlib import Path
from time import perf_counter as timer
from .toolbox.utterance import Utterance
import numpy as np
import traceback
import os
import librosa


# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

filterwarnings("ignore")


colormap = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255 

default_text = \
    "Welcome to the toolbox! To begin, load an utterance from your datasets or record one " \
    "yourself.\nOnce its embedding has been created, you can synthesize any text written here.\n" \
    "With the current synthesizer model, punctuation and special characters will be ignored.\n" \
    "The synthesizer expects to generate " \
    "outputs that are somewhere between 5 and 12 seconds.\nTo mark breaks, write a new line. " \
    "Each line will be treated separately.\nThen, they are joined together to make the final " \
    "spectrogram. Use the vocoder to generate audio.\nThe vocoder generates almost in constant " \
    "time, so it will be more time efficient for longer inputs like this one.\nOn the left you " \
    "have the embedding projections. Load or record more utterances to see them.\nIf you have " \
    "at least 2 or 3 utterances from a same speaker, a cluster should form.\nSynthesized " \
    "utterances are of the same color as the speaker whose voice was used, but they're " \
    "represented with a cross."

class VoiceCloner:
    def __init__(self):
        sys.excepthook = self.excepthook
        self.datasets_root = ""
        self.low_mem = False
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav
        
        self.synthesizer = None # type: Synthesizer
        self.current_encoder_fpath = Path('Voice_Cloning/encoder/saved_models')
        self.current_vocoder_fpath = Path("Voice_Cloning/vocoder/saved_models")
        self.current_synthesizer_fpath = Path("Voice_Cloning/synthesizer/saved_models")
        print(self.current_encoder_fpath)
        self.logs = []
        self.utterance_history = []

        #Initialize the models
        self.populate_models(self.current_encoder_fpath, self.current_synthesizer_fpath, self.current_vocoder_fpath)
        self.init_models()

        #Load the MLK files
        MLKAudioDirectory = 'Voice_Cloning/MLKAudio/'
        MLKAudioFiles = os.listdir(MLKAudioDirectory)
        for audioFile in MLKAudioFiles:
            self.load_from_browser(Path(MLKAudioDirectory + audioFile))
        

    def init_models(self):
        self.init_encoder()
        self.init_vocoder()
        
    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.log("Exception: %s" % exc_value)
        
    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))
        
        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        def func(): 
            self.synthesizer = None
        self.ui.synthesizer_box.currentIndexChanged.connect(func)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)
        
        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)
        
        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        
        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def reset_ui(self, encoder_models_dir, synthesizer_models_dir, vocoder_models_dir):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(encoder_models_dir, synthesizer_models_dir, vocoder_models_dir)
        
    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name
            
            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return 
        else:
            name = fpath.name
            speaker_name = fpath.parent.name
        
        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer.load_preprocess_wav(fpath)
        self.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)
        
    def record(self):
        wav = self.record_one(encoder.sampling_rate, 5)
        if wav is None:
            return 
        self.play(wav, encoder.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)
        
    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)

        # Compute the embedding
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.register_utterance(utterance)
        
    def clear_utterances(self):
        self.utterances.clear()
        
    def synthesize(self, text):
        print("Generating the mel spectrogram...")
        
        # Synthesize the spectrogram
        if self.synthesizer is None:
            model_dir = self.current_synthesizer_model_dir
            checkpoints_dir = model_dir.joinpath("taco_pretrained")
            self.synthesizer = Synthesizer(checkpoints_dir, low_mem=self.low_mem)
        if not self.synthesizer.is_loaded():
            print("Loading the synthesizer %s" % self.synthesizer.checkpoint_fpath)
        
        texts = text.split("\n")
        embed = self.selected_utterance.embed
        embeds = np.stack([embed] * len(texts))
        specs = self.synthesizer.synthesize_spectrograms(texts, embeds)
        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        
        self.current_generated = (self.selected_utterance.speaker_name, spec, breaks, None)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Synthesize the waveform
        if not vocoder.is_loaded():
            self.init_vocoder()
        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            print(line, "overwrite")
        if self.current_vocoder_fpath is not None:
            wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
        else:
            print("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer.griffin_lim(spec)
        print(" Done!", "append")
        
        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Play it
        wav = wav / np.abs(wav).max() * 0.97
        wav = np.pad(wav, (0, Synthesizer.sample_rate), mode="constant")
        librosa.output.write_wav("speech.wav", wav.astype(np.float32), Synthesizer.sample_rate)
        #self.play(wav, Synthesizer.sample_rate)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder.is_loaded():
            self.init_encoder()
        encoder_wav = encoder.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Add the utterance
        name = speaker_name + "_gen_%05d" % np.random.randint(100000)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)
        
    def init_encoder(self):
        model_fpath = self.current_encoder_fpath
        
        print("Loading the encoder %s... " % model_fpath)
        start = timer()
        encoder.load_model(model_fpath)
        print("Done (%dms)." % int(1000 * (timer() - start)), "append")
           
    def init_vocoder(self):
        model_fpath = self.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return 
        
        print("Loading the vocoder %s... " % model_fpath)
        start = timer()
        vocoder.load_model(model_fpath)
        print("Done (%dms)." % int(1000 * (timer() - start)), "append")

    min_umap_points = 4
    max_log_lines = 5
    max_saved_utterances = 20
    
    def play(self, wav, sample_rate):
        sd.stop()
        sd.play(wav, sample_rate)
        
    def stop(self):
        sd.stop()

    def record_one(self, sample_rate, duration):
        self.record_button.setText("Recording...")
        self.record_button.setDisabled(True)
        
        self.log("Recording %d seconds of audio" % duration)
        sd.stop()
        try:
            wav = sd.rec(duration * sample_rate, sample_rate, 1)
        except Exception as e:
            print(e)
            self.log("Could not record anything. Is your recording device enabled?")
            self.log("Your device must be connected before you start the toolbox.")
            return None
        
        for i in np.arange(0, duration, 0.1):
            self.set_loading(i, duration)
            sleep(0.1)
        self.set_loading(duration, duration)
        sd.wait()
        
        self.log("Done recording.")
        self.record_button.setText("Record one")
        self.record_button.setDisabled(False)
        
        return wav.squeeze()

    '''
    @property        
    def current_dataset_name(self):
        return self.dataset_box.currentText()

    @property
    def current_speaker_name(self):
        return self.speaker_box.currentText()
    
    @property
    def current_utterance_name(self):
        return self.utterance_box.currentText()
    '''
    
    def browse_file(self):
        fpath = QFileDialog().getOpenFileName(
            parent=self,
            caption="Select an audio file",
            filter="Audio Files (*.mp3 *.flac *.wav *.m4a)"
        )
        return Path(fpath[0]) if fpath[0] != "" else ""
    
    @staticmethod
    def repopulate_box(box, items, random=False):
        """
        Resets a box and adds a list of items. Pass a list of (item, data) pairs instead to join 
        data to the items
        """
        box.blockSignals(True)
        box.clear()
        for item in items:
            item = list(item) if isinstance(item, tuple) else [item]
            box.addItem(str(item[0]), *item[1:])
        if len(items) > 0:
            box.setCurrentIndex(np.random.randint(len(items)) if random else 0)
        box.setDisabled(len(items) == 0)
        box.blockSignals(False)
    
    def populate_browser(self, datasets_root: Path, recognized_datasets: List, level: int,
                         random=True):
        # Select a random dataset
        if level <= 0:
            if datasets_root is not None:
                datasets = [datasets_root.joinpath(d) for d in recognized_datasets]
                datasets = [d.relative_to(datasets_root) for d in datasets if d.exists()]
                self.browser_load_button.setDisabled(len(datasets) == 0)
            if datasets_root is None or len(datasets) == 0:
                msg = "Warning: you d" + ("id not pass a root directory for datasets as argument" \
                    if datasets_root is None else "o not have any of the recognized datasets" \
                                                  " in %s" % datasets_root) 
                self.log(msg)
                msg += ".\nThe recognized datasets are:\n\t%s\nFeel free to add your own. You " \
                       "can still use the toolbox by recording samples yourself." % \
                       ("\n\t".join(recognized_datasets))
                print(msg, file=sys.stderr)
                
                self.random_utterance_button.setDisabled(True)
                self.random_speaker_button.setDisabled(True)
                self.random_dataset_button.setDisabled(True)
                self.utterance_box.setDisabled(True)
                self.speaker_box.setDisabled(True)
                self.dataset_box.setDisabled(True)
                return 
            self.repopulate_box(self.dataset_box, datasets, random)
    
        # Select a random speaker
        if level <= 1:
            speakers_root = datasets_root.joinpath(self.current_dataset_name)
            speaker_names = [d.stem for d in speakers_root.glob("*") if d.is_dir()]
            self.repopulate_box(self.speaker_box, speaker_names, random)
    
        # Select a random utterance
        if level <= 2:
            utterances_root = datasets_root.joinpath(
                self.current_dataset_name, 
                self.current_speaker_name
            )
            utterances = []
            for extension in ['mp3', 'flac', 'wav', 'm4a']:
                utterances.extend(Path(utterances_root).glob("**/*.%s" % extension))
            utterances = [fpath.relative_to(utterances_root) for fpath in utterances]
            self.repopulate_box(self.utterance_box, utterances, random)
            
    def browser_select_next(self):
        index = (self.utterance_box.currentIndex() + 1) % len(self.utterance_box)
        self.utterance_box.setCurrentIndex(index)

    '''
    @property
    def current_encoder_fpath(self):
        return self.encoder_box.itemData(self.encoder_box.currentIndex())
    
    @property
    def current_synthesizer_model_dir(self):
        return self.synthesizer_box.itemData(self.synthesizer_box.currentIndex())
    
    @property
    def current_vocoder_fpath(self):
        return self.vocoder_box.itemData(self.vocoder_box.currentIndex())
    '''

    def populate_models(self, encoder_models_dir: Path, synthesizer_models_dir: Path, 
                        vocoder_models_dir: Path):
        # Encoder
        encoder_fpaths = list(encoder_models_dir.glob("*.pt"))
        if len(encoder_fpaths) == 0:
            raise Exception("No encoder models found in %s" % encoder_models_dir)
        self.current_encoder_fpath = encoder_fpaths[0]
        
        # Synthesizer
        synthesizer_model_dirs = list(synthesizer_models_dir.glob("*"))
        synthesizer_items = [(f.name.replace("logs-", ""), f) for f in synthesizer_model_dirs]
        if len(synthesizer_model_dirs) == 0:
            raise Exception("No synthesizer models found in %s. For the synthesizer, the expected "
                            "structure is <syn_models_dir>/logs-<model_name>/taco_pretrained/"
                            "checkpoint" % synthesizer_models_dir)
        self.current_synthesizer_model_dir = synthesizer_model_dirs[0]

        # Vocoder
        vocoder_fpaths = list(vocoder_models_dir.glob("**/*.pt"))
        vocoder_items = [(f.stem, f) for f in vocoder_fpaths] + [("Griffin-Lim", None)]
        self.current_vocoder_fpath = vocoder_fpaths[0]
        
    def register_utterance(self, utterance: Utterance):
        self.utterance_history.append(utterance)
        self.selected_utterance = utterance

    def log(self, line, mode="newline"):
        if mode == "newline":
            self.logs.append(line)
            if len(self.logs) > self.max_log_lines:
                del self.logs[0]
        elif mode == "append":
            self.logs[-1] += line
        elif mode == "overwrite":
            self.logs[-1] = line
        log_text = '\n'.join(self.logs)
        
        print(log_text)
        
    '''
    def reset_interface(self):
        self.draw_embed(None, None, "current")
        self.draw_embed(None, None, "generated")
        self.draw_spec(None, "current")
        self.draw_spec(None, "generated")
        self.draw_umap_projections(set())
        self.set_loading(0)
        self.play_button.setDisabled(True)
        self.generate_button.setDisabled(True)
        self.synthesize_button.setDisabled(True)
        self.vocode_button.setDisabled(True)
        [self.log("") for _ in range(self.max_log_lines)]
        
    def __init__(self):
        ## Initialize the application
        self.app = QApplication(sys.argv)
        super().__init__(None)
        self.setWindowTitle("SV2TTS toolbox")
        
        
        ## Main layouts
        # Root
        root_layout = QGridLayout()
        self.setLayout(root_layout)
        
        # Browser
        browser_layout = QGridLayout()
        root_layout.addLayout(browser_layout, 0, 1)
        
        # Visualizations
        vis_layout = QVBoxLayout()
        root_layout.addLayout(vis_layout, 1, 1, 2, 3)
        
        # Generation
        gen_layout = QVBoxLayout()
        root_layout.addLayout(gen_layout, 0, 2)
        
        # Projections
        self.projections_layout = QVBoxLayout()
        root_layout.addLayout(self.projections_layout, 1, 0)


        ## Projections
        # UMap
        fig, self.umap_ax = plt.subplots(figsize=(4, 4), facecolor="#F0F0F0")
        fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
        self.projections_layout.addWidget(FigureCanvas(fig))
        self.umap_hot = False
        self.clear_button = QPushButton("Clear")
        self.projections_layout.addWidget(self.clear_button)


        ## Browser
        # Dataset, speaker and utterance selection
        i = 0
        self.dataset_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Dataset</b>"), i, 0)
        browser_layout.addWidget(self.dataset_box, i + 1, 0)
        self.speaker_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Speaker</b>"), i, 1)
        browser_layout.addWidget(self.speaker_box, i + 1, 1)
        self.utterance_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Utterance</b>"), i, 2)
        browser_layout.addWidget(self.utterance_box, i + 1, 2)
        self.browser_browse_button = QPushButton("Browse")
        browser_layout.addWidget(self.browser_browse_button, i, 3)
        self.browser_load_button = QPushButton("Load")
        browser_layout.addWidget(self.browser_load_button, i + 1, 3)
        i += 2
        
        # Random buttons
        self.random_dataset_button = QPushButton("Random")
        browser_layout.addWidget(self.random_dataset_button, i, 0)
        self.random_speaker_button = QPushButton("Random")
        browser_layout.addWidget(self.random_speaker_button, i, 1)
        self.random_utterance_button = QPushButton("Random")
        browser_layout.addWidget(self.random_utterance_button, i, 2)
        self.auto_next_checkbox = QCheckBox("Auto select next")
        self.auto_next_checkbox.setChecked(True)
        browser_layout.addWidget(self.auto_next_checkbox, i, 3)
        i += 1
        
        # Utterance box
        browser_layout.addWidget(QLabel("<b>Use embedding from:</b>"), i, 0)
        i += 1
        
        # Random & next utterance buttons
        self.utterance_history = QComboBox()
        browser_layout.addWidget(self.utterance_history, i, 0, 1, 3)
        i += 1
        
        # Random & next utterance buttons
        self.take_generated_button = QPushButton("Take generated")
        browser_layout.addWidget(self.take_generated_button, i, 0)
        self.record_button = QPushButton("Record")
        browser_layout.addWidget(self.record_button, i, 1)
        self.play_button = QPushButton("Play")
        browser_layout.addWidget(self.play_button, i, 2)
        self.stop_button = QPushButton("Stop")
        browser_layout.addWidget(self.stop_button, i, 3)
        i += 2

        # Model selection
        self.encoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Encoder</b>"), i, 0)
        browser_layout.addWidget(self.encoder_box, i + 1, 0)
        self.synthesizer_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Synthesizer</b>"), i, 1)
        browser_layout.addWidget(self.synthesizer_box, i + 1, 1)
        self.vocoder_box = QComboBox()
        browser_layout.addWidget(QLabel("<b>Vocoder</b>"), i, 2)
        browser_layout.addWidget(self.vocoder_box, i + 1, 2)
        i += 2


        ## Embed & spectrograms
        vis_layout.addStretch()

        gridspec_kw = {"width_ratios": [1, 4]}
        fig, self.current_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#F0F0F0", 
                                            gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        fig, self.gen_ax = plt.subplots(1, 2, figsize=(10, 2.25), facecolor="#F0F0F0", 
                                        gridspec_kw=gridspec_kw)
        fig.subplots_adjust(left=0, bottom=0.1, right=1, top=0.8)
        vis_layout.addWidget(FigureCanvas(fig))

        for ax in self.current_ax.tolist() + self.gen_ax.tolist():
            ax.set_facecolor("#F0F0F0")
            for side in ["top", "right", "bottom", "left"]:
                ax.spines[side].set_visible(False)
        
        
        ## Generation
        self.text_prompt = QPlainTextEdit(default_text)
        gen_layout.addWidget(self.text_prompt, stretch=1)
        
        self.generate_button = QPushButton("Synthesize and vocode")
        gen_layout.addWidget(self.generate_button)
        
        layout = QHBoxLayout()
        self.synthesize_button = QPushButton("Synthesize only")
        layout.addWidget(self.synthesize_button)
        self.vocode_button = QPushButton("Vocode only")
        layout.addWidget(self.vocode_button)
        gen_layout.addLayout(layout)

        self.loading_bar = QProgressBar()
        gen_layout.addWidget(self.loading_bar)
        
        self.log_window = QLabel()
        self.log_window.setAlignment(Qt.AlignBottom | Qt.AlignLeft)
        gen_layout.addWidget(self.log_window)
        self.logs = []
        gen_layout.addStretch()

        
        ## Set the size of the window and of the elements
        max_size = QDesktopWidget().availableGeometry(self).size() * 0.8
        self.resize(max_size)
        
        ## Finalize the display
        self.reset_interface()
        self.show()

    def start(self):
        self.app.exec_()
    '''
