"""
Multiple utilities
"""
import yt_dlp
from mt3_utils import get_mt3_model, load_audio, save_seq_to_midi
from figaro_utils import get_description_from_midi_path

midi_transcriber = get_mt3_model()

def youtube_ids(file_path="data/Youtube_ID.txt"):
    with open(file_path) as f:
        for line in f.readlines():
            youtube_id = line.strip().split("=")[-1]
            yield youtube_id


def youtube_urls(file_path="data/Youtube_ID.txt"):
    with open(file_path) as f:
        for line in f.readlines():
            youtube_url = line.strip()
            yield youtube_url



# For more details check: https://github.com/yt-dlp/yt-dlp#embedding-examples

# ydl_opts = {
#     # 'format': 'm4a/bestaudio/best',
#     # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
#     # 'postprocessors': [{  # Extract audio using ffmpeg
#     #     'key': 'FFmpegExtractAudio',
#     #     'preferredcodec': 'm4a',
#     # }]
#     #  'postprocessors': [{  # Extract audio using ffmpeg
#     #     'key': 'FFmpegVideoConvertor',
#     #     'preferedformat': 'mp4',
#     # }],

#     # TODO: Add the following post processors
#     # MoveFilesAfterDownloadPP
#     # FFmpegVideoRemuxerPP

#     "concurrent_fragments": 100,



#     # 'merge-output-format': "mp4"


# }

# def format_selector(ctx):
#     """ Select the best video and the best audio that won't result in an mkv.
#     NOTE: This is just an example and does not handle all cases """

#     # formats are already sorted worst to best
#     formats = ctx.get('formats')[::-1]

#     # acodec='none' means there is no audio
#     best_video = next(f for f in formats
#                       if f['vcodec'] != 'none' and f['acodec'] == 'none')

#     # find compatible audio extension
#     audio_ext = {'mp4': 'm4a', 'webm': 'webm'}[best_video['ext']]
#     # vcodec='none' means there is no video
#     best_audio = next(f for f in formats if (
#         f['acodec'] != 'none' and f['vcodec'] == 'none' and f['ext'] == audio_ext))

#     # These are the minimum required fields for a merged format
#     yield {
#         'format_id': f'{best_video["format_id"]}+{best_audio["format_id"]}',
#         'ext': best_video['ext'],
#         'requested_formats': [best_video, best_audio],
#         # Must be + separated list of protocols
#         'protocol': f'{best_video["protocol"]}+{best_audio["protocol"]}'
#     }

# class MyCustomPP(yt_dlp.postprocessor.PostProcessor):
#     def run(self, info):
#         self.to_screen('Doing stuff')
#         return [], info

class TranscribeMIDI(yt_dlp.postprocessor.PostProcessor):
    def run(self, information):
        information['ext'] = 'mid'
        orig_path = information['filepath']

        audio = load_audio(orig_path)

        midi_like_sequence = midi_transcriber(audio)

        orig_no_ext = ".".join(orig_path.split(".")[:-1])

        target_path = orig_no_ext + ".mid"

        save_seq_to_midi(midi_like_sequence, target_path)

        return [target_path], information
    
class FigaroDescription(yt_dlp.postprocessor.PostProcessor):
    def run(self, information):
        information['ext'] = 'desc'
        orig_path = information['filepath']

        description = get_description_from_midi_path(orig_path)

        print('FIGARO DESCRIPTION', description)

        orig_no_ext = ".".join(orig_path.split(".")[:-1])

        target_path = orig_no_ext + ".desc"

        with open(target_path, 'a') as f:
            f.write(description)

        return [target_path], information


def download_video(youtube_url, output_dir="./data/videos"):
    ydl_opts = {
        'format': 'best',
        "concurrent_fragments": 100,
        'extractaudio': True,
        'audioformat': 'mp3',
        'keepvideo': True,
        'remuxvideo': 'mp4',
        "paths": {
            'home': output_dir,
            'temp': "tmp"
            },
        "outtmpl": "%(id)s.%(ext)s",
        # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
        #  'postprocessors': [{  # Extract audio using ffmpeg
        #     'key': 'FFmpegVideoRemuxer',
        #     'preferedformat': 'mp4',
        # }],

        # TODO: Add the following post processors
        # MoveFilesAfterDownloadPP
        # FFmpegVideoRemuxerPP
    }
    print(ydl_opts)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.add_post_processor(TranscribeMIDI())
        ydl.add_post_processor(FigaroDescription())
        ydl.download([youtube_url])

# def download_videos(urls):
#     print(ydl_opts)
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download(urls)
