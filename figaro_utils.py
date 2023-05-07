import sys

# sys.path.append('figaro/src')
# from datasets import MidiDataset
import numpy as np
import pretty_midi
import numpy as np
from figaro_chords import MIDIChord

class Item(object):
  def __init__(self, name, start, end, velocity=None, pitch=None, instrument=None):
    self.name = name
    self.start = start
    self.end = end
    self.velocity = velocity
    self.pitch = pitch
    self.instrument = instrument

  def __repr__(self):
    return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, instrument={})'.format(
      self.name, self.start, self.end, self.velocity, self.pitch, self.instrument)

# define "Event" for event storage
class Event(object):
  def __init__(self, name, time, value, text):
    self.name = name
    self.time = time
    self.value = value
    self.text = text

  def __repr__(self):
    return 'Event(name={}, time={}, value={}, text={})'.format(
      self.name, self.time, self.value, self.text)

# parameters for input representation
DEFAULT_POS_PER_QUARTER = 12
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_DURATION_BINS = np.sort(np.concatenate([
  np.arange(1, 13), # smallest possible units up to 1 quarter
  np.arange(12, 24, 3)[1:], # 16th notes up to 1 bar
  np.arange(13, 24, 4)[1:], # triplets up to 1 bar
  np.arange(24, 48, 6), # 8th notes up to 2 bars
  np.arange(48, 4*48, 12), # quarter notes up to 8 bars
  np.arange(4*48, 16*48+1, 24) # half notes up to 16 bars
]))
DEFAULT_TEMPO_BINS = np.linspace(0, 240, 32+1, dtype=np.int)
DEFAULT_NOTE_DENSITY_BINS = np.linspace(0, 12, 32+1)
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 32+1)
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 128, 32+1)
DEFAULT_MEAN_DURATION_BINS = np.logspace(0, 7, 32+1, base=2) # log space between 1 and 128 positions (~2.5 bars)

# parameters for output
DEFAULT_RESOLUTION = 480

# maximum length of a single bar is 3*4 = 12 beats
MAX_BAR_LENGTH = 3
# maximum number of bars in a piece is 512 (this covers almost all sequences)
MAX_N_BARS = 512

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
MASK_TOKEN = '<mask>'

TIME_SIGNATURE_KEY = 'Time Signature'
BAR_KEY = 'Bar'
POSITION_KEY = 'Position'
INSTRUMENT_KEY = 'Instrument'
PITCH_KEY = 'Pitch'
VELOCITY_KEY = 'Velocity'
DURATION_KEY = 'Duration'
TEMPO_KEY = 'Tempo'
CHORD_KEY = 'Chord'

NOTE_DENSITY_KEY = 'Note Density'
MEAN_PITCH_KEY = 'Mean Pitch'
MEAN_VELOCITY_KEY = 'Mean Velocity'
MEAN_DURATION_KEY = 'Mean Duration'


class InputRepresentation():
  def version():
    return 'v4'
  
  def __init__(self, file, do_extract_chords=True, strict=False):
    if isinstance(file, pretty_midi.PrettyMIDI):
      self.pm = file
    else:
      self.pm = pretty_midi.PrettyMIDI(file)

    if strict and len(self.pm.time_signature_changes) == 0:
      raise ValueError("Invalid MIDI file: No time signature defined")

    self.resolution = self.pm.resolution

    self.note_items = None
    self.tempo_items = None
    self.chords = None
    self.groups = None
    
    self._read_items()
    self._quantize_items()
    if do_extract_chords:
      self.extract_chords()
    self._group_items()

    if strict and len(self.note_items) == 0:
      raise ValueError("Invalid MIDI file: No notes found, empty file.")

  # read notes and tempo changes from midi (assume there is only one track)
  def _read_items(self):
    # note
    self.note_items = []
    for instrument in self.pm.instruments:
      pedal_events = [event for event in instrument.control_changes if event.number == 64]
      pedal_pressed = False
      start = None
      pedals = []
      for e in pedal_events:
        if e.value >= 64 and not pedal_pressed:
          pedal_pressed = True
          start = e.time
        elif e.value < 64 and pedal_pressed:
          pedal_pressed = False
          pedals.append(Item(name='Pedal', start=start, end=e.time))
          start = e.time

      notes = instrument.notes
      notes.sort(key=lambda x: (x.start, x.pitch))

      if instrument.is_drum:
        instrument_name = 'drum'
      else:
        instrument_name = instrument.program
      
      pedal_idx = 0
      for note in notes:
        pedal_candidates = [(i + pedal_idx, pedal) for i, pedal in enumerate(pedals[pedal_idx:]) if note.end >= pedal.start and note.start < pedal.end]
        if len(pedal_candidates) > 0:
          pedal_idx = pedal_candidates[0][0]
          pedal = pedal_candidates[-1][1]
        else:
          pedal = Item(name='Pedal', start=0, end=0)
        
        self.note_items.append(Item(
          name='Note', 
          start=self.pm.time_to_tick(note.start), 
          end=self.pm.time_to_tick(max(note.end, pedal.end)), 
          velocity=note.velocity, 
          pitch=note.pitch,
          instrument=instrument_name))
    self.note_items.sort(key=lambda x: (x.start, x.pitch))
    # tempo
    self.tempo_items = []
    times, tempi = self.pm.get_tempo_changes()
    for time, tempo in zip(times, tempi):
      self.tempo_items.append(Item(
        name='Tempo',
        start=time,
        end=None,
        velocity=None,
        pitch=int(tempo)))
    self.tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = self.pm.time_to_tick(self.pm.get_end_time())
    existing_ticks = {item.start: item.pitch for item in self.tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
      if tick in existing_ticks:
        output.append(Item(
          name='Tempo',
          start=self.pm.time_to_tick(tick),
          end=None,
          velocity=None,
          pitch=existing_ticks[tick]))
      else:
        output.append(Item(
          name='Tempo',
          start=self.pm.time_to_tick(tick),
          end=None,
          velocity=None,
          pitch=output[-1].pitch))
    self.tempo_items = output

  # quantize items
  def _quantize_items(self):
    ticks = self.resolution / DEFAULT_POS_PER_QUARTER
    # grid
    end_tick = self.pm.time_to_tick(self.pm.get_end_time())
    grids = np.arange(0, max(self.resolution, end_tick), ticks)
    # process
    for item in self.note_items:
      index = np.searchsorted(grids, item.start, side='right')
      if index > 0:
        index -= 1
      shift = round(grids[index]) - item.start
      item.start += shift
      item.end += shift

  def get_end_tick(self):
    return self.pm.time_to_tick(self.pm.get_end_time())

  # extract chord
  def extract_chords(self):
    end_tick = self.pm.time_to_tick(self.pm.get_end_time())
    if end_tick < self.resolution:
      # If sequence is shorter than 1/4th note, it's probably empty
      self.chords = []
      return self.chords
    method = MIDIChord(self.pm)
    chords = method.extract()
    output = []
    for chord in chords:
      output.append(Item(
        name='Chord',
        start=self.pm.time_to_tick(chord[0]),
        end=self.pm.time_to_tick(chord[1]),
        velocity=None,
        pitch=chord[2].split('/')[0]))
    if len(output) == 0 or output[0].start > 0:
      if len(output) == 0:
        end = self.pm.time_to_tick(self.pm.get_end_time())
      else:
        end = output[0].start
      output.append(Item(
        name='Chord',
        start=0,
        end=end,
        velocity=None,
        pitch='N:N'
      ))
    self.chords = output
    return self.chords

  # group items
  def _group_items(self):
    if self.chords:
      items = self.chords + self.tempo_items + self.note_items
    else:
      items = self.tempo_items + self.note_items

    def _get_key(item):
      type_priority = {
        'Chord': 0,
        'Tempo': 1,
        'Note': 2
      }
      return (
        item.start, # order by time
        type_priority[item.name], # chord events first, then tempo events, then note events
        -1 if item.instrument == 'drum' else item.instrument, # order by instrument
        item.pitch # order by note pitch
      )

    items.sort(key=_get_key)
    downbeats = self.pm.get_downbeats()
    downbeats = np.concatenate([downbeats, [self.pm.get_end_time()]])
    self.groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
      db1, db2 = self.pm.time_to_tick(db1), self.pm.time_to_tick(db2)
      insiders = []
      for item in items:
        if (item.start >= db1) and (item.start < db2):
          insiders.append(item)
      overall = [db1] + insiders + [db2]
      self.groups.append(overall)

    # Trim empty groups from the beginning and end
    for idx in [0, -1]:
      while len(self.groups) > 0:
        group = self.groups[idx]
        notes = [item for item in group[1:-1] if item.name == 'Note']
        if len(notes) == 0:
          self.groups.pop(idx)
        else:
          break
    
    return self.groups
  
  def _get_time_signature(self, start):
    # This method assumes that time signature changes don't happen within a bar
    # which is a convention that commonly holds
    time_sig = None
    for curr_sig, next_sig in zip(self.pm.time_signature_changes[:-1], self.pm.time_signature_changes[1:]):
      if self.pm.time_to_tick(curr_sig.time) <= start and self.pm.time_to_tick(next_sig.time) > start:
        time_sig = curr_sig
        break
    if time_sig is None:
      time_sig = self.pm.time_signature_changes[-1]
    return time_sig

  def _get_ticks_per_bar(self, start):
    time_sig = self._get_time_signature(start)
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
    return self.pm.resolution * quarters_per_bar

  def _get_positions_per_bar(self, start=None, time_sig=None):
    if time_sig is None:
      time_sig = self._get_time_signature(start)
    quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
    positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)
    return positions_per_bar
  
  def tick_to_position(self, tick):
    return round(tick / self.pm.resolution * DEFAULT_POS_PER_QUARTER)

  # item to event
#   def get_remi_events(self):
#     events = []
#     n_downbeat = 0
#     current_chord = None
#     current_tempo = None
#     for i in range(len(self.groups)):
#       bar_st, bar_et = self.groups[i][0], self.groups[i][-1]
#       n_downbeat += 1
#       positions_per_bar = self._get_positions_per_bar(bar_st)
#       if positions_per_bar <= 0:
#         raise ValueError('Invalid REMI file: There must be at least 1 position per bar.')

#       events.append(Event(
#         name=BAR_KEY,
#         time=None, 
#         value='{}'.format(n_downbeat),
#         text='{}'.format(n_downbeat)))

#       time_sig = self._get_time_signature(bar_st)
#       events.append(Event(
#         name=TIME_SIGNATURE_KEY,
#         time=None,
#         value='{}/{}'.format(time_sig.numerator, time_sig.denominator),
#         text='{}/{}'.format(time_sig.numerator, time_sig.denominator)
#       ))

#       if current_chord is not None:
#         events.append(Event(
#           name=POSITION_KEY, 
#           time=0,
#           value='{}'.format(0),
#           text='{}/{}'.format(1, positions_per_bar)))
#         events.append(Event(
#           name=CHORD_KEY,
#           time=current_chord.start,
#           value=current_chord.pitch,
#           text='{}'.format(current_chord.pitch)))
      
#       if current_tempo is not None:
#         events.append(Event(
#           name=POSITION_KEY, 
#           time=0,
#           value='{}'.format(0),
#           text='{}/{}'.format(1, positions_per_bar)))
#         tempo = current_tempo.pitch
#         index = np.argmin(abs(DEFAULT_TEMPO_BINS-tempo))
#         events.append(Event(
#           name=TEMPO_KEY,
#           time=current_tempo.start,
#           value=index,
#           text='{}/{}'.format(tempo, DEFAULT_TEMPO_BINS[index])))
      
#       quarters_per_bar = 4 * time_sig.numerator / time_sig.denominator
#       ticks_per_bar = self.pm.resolution * quarters_per_bar
#       flags = np.linspace(bar_st, bar_st + ticks_per_bar, positions_per_bar, endpoint=False)
#       for item in self.groups[i][1:-1]:
#         # position
#         index = np.argmin(abs(flags-item.start))
#         pos_event = Event(
#           name=POSITION_KEY, 
#           time=item.start,
#           value='{}'.format(index),
#           text='{}/{}'.format(index+1, positions_per_bar))

#         if item.name == 'Note':
#           events.append(pos_event)
#           # instrument
#           if item.instrument == 'drum':
#             name = 'drum'
#           else:
#             name = pretty_midi.program_to_instrument_name(item.instrument)
#           events.append(Event(
#             name=INSTRUMENT_KEY,
#             time=item.start, 
#             value=name,
#             text='{}'.format(name)))
#           # pitch
#           events.append(Event(
#             name=PITCH_KEY,
#             time=item.start, 
#             value='drum_{}'.format(item.pitch) if name == 'drum' else item.pitch,
#             text='{}'.format(pretty_midi.note_number_to_name(item.pitch))))
#           # velocity
#           velocity_index = np.argmin(abs(DEFAULT_VELOCITY_BINS - item.velocity))
#           events.append(Event(
#             name=VELOCITY_KEY,
#             time=item.start, 
#             value=velocity_index,
#             text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
#           # duration
#           duration = self.tick_to_position(item.end - item.start)
#           index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
#           events.append(Event(
#             name=DURATION_KEY,
#             time=item.start,
#             value=index,
#             text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
#         elif item.name == 'Chord':
#           if current_chord is None or item.pitch != current_chord.pitch:
#             events.append(pos_event)
#             events.append(Event(
#               name=CHORD_KEY, 
#               time=item.start,
#               value=item.pitch,
#               text='{}'.format(item.pitch)))
#             current_chord = item
#         elif item.name == 'Tempo':
#           if current_tempo is None or item.pitch != current_tempo.pitch:
#             events.append(pos_event)
#             tempo = item.pitch
#             index = np.argmin(abs(DEFAULT_TEMPO_BINS-tempo))
#             events.append(Event(
#               name=TEMPO_KEY,
#               time=item.start,
#               value=index,
#               text='{}/{}'.format(tempo, DEFAULT_TEMPO_BINS[index])))
#             current_tempo = item
    
#     return [f'{e.name}_{e.value}' for e in events]

  def get_description(self, 
                      omit_time_sig=False,
                      omit_instruments=False,
                      omit_chords=False,
                      omit_meta=False):
    events = []
    n_downbeat = 0
    current_chord = None

    for i in range(len(self.groups)):
      bar_st, bar_et = self.groups[i][0], self.groups[i][-1]
      n_downbeat += 1
      time_sig = self._get_time_signature(bar_st)
      positions_per_bar = self._get_positions_per_bar(time_sig=time_sig)
      if positions_per_bar <= 0:
        raise ValueError('Invalid REMI file: There must be at least 1 position in each bar.')

      events.append(Event(
        name=BAR_KEY,
        time=None, 
        value='{}'.format(n_downbeat),
        text='{}'.format(n_downbeat)))
      
      if not omit_time_sig:
        events.append(Event(
          name=TIME_SIGNATURE_KEY,
          time=None,
          value='{}/{}'.format(time_sig.numerator, time_sig.denominator),
          text='{}/{}'.format(time_sig.numerator, time_sig.denominator),
        ))

      if not omit_meta:
        notes = [item for item in self.groups[i][1:-1] if item.name == 'Note']
        n_notes = len(notes)
        velocities = np.array([item.velocity for item in notes])
        pitches = np.array([item.pitch for item in notes])
        durations = np.array([item.end - item.start for item in notes])

        note_density = n_notes/positions_per_bar
        index = np.argmin(abs(DEFAULT_NOTE_DENSITY_BINS-note_density))
        events.append(Event(
          name=NOTE_DENSITY_KEY,
          time=None,
          value=index,
          text='{:.2f}/{:.2f}'.format(note_density, DEFAULT_NOTE_DENSITY_BINS[index])
        ))

        # will be 0 if there's no notes
        mean_velocity = velocities.mean() if len(velocities) > 0 else np.nan
        index = np.argmin(abs(DEFAULT_MEAN_VELOCITY_BINS-mean_velocity))
        events.append(Event(
          name=MEAN_VELOCITY_KEY,
          time=None,
          value=index if mean_velocity != np.nan else 'NaN',
          text='{:.2f}/{:.2f}'.format(mean_velocity, DEFAULT_MEAN_VELOCITY_BINS[index])
        ))

        # will be 0 if there's no notes
        mean_pitch = pitches.mean() if len(pitches) > 0 else np.nan
        index = np.argmin(abs(DEFAULT_MEAN_PITCH_BINS-mean_pitch))
        events.append(Event(
          name=MEAN_PITCH_KEY,
          time=None,
          value=index if mean_pitch != np.nan else 'NaN',
          text='{:.2f}/{:.2f}'.format(mean_pitch, DEFAULT_MEAN_PITCH_BINS[index])
        ))

        # will be 1 if there's no notes
        mean_duration = durations.mean() if len(durations) > 0 else np.nan
        index = np.argmin(abs(DEFAULT_MEAN_DURATION_BINS-mean_duration))
        events.append(Event(
          name=MEAN_DURATION_KEY,
          time=None,
          value=index if mean_duration != np.nan else 'NaN',
          text='{:.2f}/{:.2f}'.format(mean_duration, DEFAULT_MEAN_DURATION_BINS[index])
        ))

      if not omit_instruments:
        instruments = set([item.instrument for item in notes])
        for instrument in instruments:
          instrument = pretty_midi.program_to_instrument_name(instrument) if instrument != 'drum' else 'drum'
          events.append(Event(
            name=INSTRUMENT_KEY,
            time=None,
            value=instrument,
            text=instrument
          ))

      if not omit_chords:
        chords = [item for item in self.groups[i][1:-1] if item.name == 'Chord']
        if len(chords) == 0 and current_chord is not None:
          chords = [current_chord]
        elif len(chords) > 0:
          if chords[0].start > bar_st and current_chord is not None:
            chords.insert(0, current_chord)
          current_chord = chords[-1]

        for chord in chords:
          events.append(Event(
            name=CHORD_KEY, 
            time=None,
            value=chord.pitch,
            text='{}'.format(chord.pitch)
          ))
        
    return [f'{e.name}_{e.value}' for e in events]


def get_description_from_midi_path(path):
    return InputRepresentation(path).get_description()
